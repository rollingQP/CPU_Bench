import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.time.Instant;
import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.util.concurrent.TimeUnit;

public class NtpTimeContinuous {
    private static final int NTP_PORT = 123;
    private static final long NTP_TIMESTAMP_DELTA = 2208988800L; // 1900->1970 秒差
    private static final byte NTP_MODE_CLIENT_V4 = 0x23; // LI=0,VN=4,Mode=3
    private static final boolean USE_ANSI_CLEAR = true;
    private static final String CR_CLEAR = "\r\u001B[2K";
    private static final int LINE_WIDTH = 140;

    // 缓存（仅在对时/偶发检查时更新）
    private static volatile double lastOffsetSec = 0.0;
    private static volatile double lastRttMs = Double.NaN;
    private static volatile long lastSyncMonoNs = 0L;
    private static volatile int zoneOffsetMillis = 0; // 本地时区相对UTC的偏移（含DST），对时后更新
    private static final ZoneId LOCAL_ZONE = ZoneId.systemDefault();

    public static void main(String[] args) {
        String server = args.length > 0 ? args[0] : "stdtime.gov.hk";
        long intervalMillis = args.length > 1 ? Long.parseLong(args[1]) : 100; // 刷新更平滑
        int timeoutMillis = 1200;

        // 启动时做一次时区偏移计算
        recalcZoneOffset(System.currentTimeMillis());

        printSingleLine("初始化中... NTP: " + server);

        while (true) {
            long loopStartNs = System.nanoTime();
            try {
                boolean needResync = (loopStartNs - lastSyncMonoNs) >= TimeUnit.SECONDS.toNanos(5);
                if (needResync) {
                    Result r = queryNtp(server, timeoutMillis);
                    lastOffsetSec = r.offsetSeconds;
                    lastRttMs = r.rttMillis;
                    lastSyncMonoNs = loopStartNs;

                    // 对时后用新的UTC毫秒重新计算本地时区偏移（包括DST）
                    long correctedNowMs = correctedNowMillis();
                    recalcZoneOffset(correctedNowMs);
                } else {
                    // 每秒检查一次是否跨越DST边界（低成本）
                    if ((System.currentTimeMillis() / 1000) % 1 == 0) {
                        long correctedNowMs = correctedNowMillis();
                        maybeUpdateZoneOffset(correctedNowMs);
                    }
                }

                // 渲染：纯数字计算+轻量拼接，不创建日期对象
                long utcMs = correctedNowMillis();
                long localMs = utcMs + zoneOffsetMillis;

                String utcStr = formatUtc(utcMs);
                String localStr = formatLocal(localMs, zoneOffsetMillis);

                String stat = String.format("RTT≈%.1fms, 偏移≈%.1fms, 来自: %s",
                        lastRttMs, lastOffsetSec * 1000.0, server);

                String line = "UTC: " + utcStr + " | Local: " + localStr + " | " + stat;
                printSingleLine(truncate(line, LINE_WIDTH));
            } catch (Exception e) {
                printSingleLine(truncate("获取NTP时间失败: " + e.getMessage(), LINE_WIDTH));
                sleepMillis(300);
            }

            long elapsedMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - loopStartNs);
            long sleep = Math.max(0, intervalMillis - elapsedMs);
            sleepMillis(sleep);
        }
    }

    // 计算“校正后的当前UTC毫秒”（使用最近一次offset）
    private static long correctedNowMillis() {
        double correctedNowUnix = (System.currentTimeMillis() / 1000.0) + lastOffsetSec;
        return (long) Math.floor(correctedNowUnix * 1000.0);
    }

    // 仅对时或检测到变化时更新本地时区偏移（含DST）
    private static void recalcZoneOffset(long utcMs) {
        // 通过Java的时区规则算一次，然后缓存为固定的毫秒偏移
        int off = ZonedDateTime.ofInstant(Instant.ofEpochMilli(utcMs), LOCAL_ZONE)
                .getOffset().getTotalSeconds() * 1000;
        zoneOffsetMillis = off;
    }
    private static void maybeUpdateZoneOffset(long utcMs) {
        int off = ZonedDateTime.ofInstant(Instant.ofEpochMilli(utcMs), LOCAL_ZONE)
                .getOffset().getTotalSeconds() * 1000;
        if (off != zoneOffsetMillis) zoneOffsetMillis = off;
    }

    // 零分配UTC格式化：yyyy-MM-dd HH:mm:ss.SSS UTC
    private static String formatUtc(long utcMs) {
        YMDHMS y = breakDownUtc(utcMs);
        StringBuilder sb = new StringBuilder(28);
        append4(sb, y.year).append('-'); append2(sb, y.month).append('-'); append2(sb, y.day).append(' ');
        append2(sb, y.hour).append(':'); append2(sb, y.min).append(':'); append2(sb, y.sec).append('.');
        append3(sb, y.millis).append(" UTC");
        return sb.toString();
    }

    // 零分配Local格式化：yyyy-MM-dd HH:mm:ss.SSS ZZZ（仅显示时区缩写/总偏移）
    private static String formatLocal(long localMs, int offsetMillis) {
        YMDHMS y = breakDownUtc(localMs); // 这里传的是“本地毫秒”，但我们用UTC拆分算法，对应于已经加上偏移的时间点
        String zoneStr = offsetToString(offsetMillis);
        StringBuilder sb = new StringBuilder(34);
        append4(sb, y.year).append('-'); append2(sb, y.month).append('-'); append2(sb, y.day).append(' ');
        append2(sb, y.hour).append(':'); append2(sb, y.min).append(':'); append2(sb, y.sec).append('.');
        append3(sb, y.millis).append(' ').append(zoneStr);
        return sb.toString();
    }

    // 把毫秒分解为UTC下的年月日时分秒毫秒（基于Unix epoch，算法无分配）
    private static YMDHMS breakDownUtc(long ms) {
        long totalSeconds = Math.floorDiv(ms, 1000L);
        int millis = (int) Math.floorMod(ms, 1000L);
        // civil date from days since epoch (1970-01-01)
        long days = Math.floorDiv(totalSeconds, 86400L);
        int secOfDay = (int) Math.floorMod(totalSeconds, 86400L);
        int hour = secOfDay / 3600;
        int min = (secOfDay % 3600) / 60;
        int sec = secOfDay % 60;

        int[] ymd = civilFromDays(days);
        YMDHMS out = new YMDHMS();
        out.year = ymd[0];
        out.month = ymd[1];
        out.day = ymd[2];
        out.hour = hour;
        out.min = min;
        out.sec = sec;
        out.millis = millis;
        return out;
    }

    // 算法参考 Howard Hinnant 的 civil_from_days（公历，无时区）
    private static int[] civilFromDays(long z) {
        z += 719468; // shift to Civil 0000-03-01 base
        long era = (z >= 0 ? z : z - 146096) / 146097;
        long doe = z - era * 146097;                    // [0, 146096]
        long yoe = (doe - doe/1460 + doe/36524 - doe/146096) / 365; // [0, 399]
        int y = (int)(yoe + era * 400);
        long doy = doe - (365*yoe + yoe/4 - yoe/100 + yoe/400);     // [0, 365]
        long mp = (5*doy + 2) / 153;                   // [0, 11]
        int d = (int)(doy - (153*mp + 2)/5 + 1);       // [1, 31]
        int m = (int)(mp + (mp < 10 ? 3 : -9));        // [1, 12]
        y += (m <= 2) ? 1 : 0;
        return new int[]{y, m, d};
    }

    private static String offsetToString(int offsetMillis) {
        // 例如 +08:00、-05:00
        int totalMinutes = Math.abs(offsetMillis) / 60000;
        int h = totalMinutes / 60;
        int m = totalMinutes % 60;
        char sign = offsetMillis >= 0 ? '+' : '-';
        return new StringBuilder(6).append(sign).append(pad2(h)).append(':').append(pad2(m)).toString();
    }

    private static String pad2(int v) { return (v < 10 ? "0" : "") + v; }

    private static StringBuilder append2(StringBuilder sb, int v) {
        if (v >= 10) return sb.append(v);
        return sb.append('0').append(v);
    }
    private static StringBuilder append3(StringBuilder sb, int v) {
        if (v >= 100) return sb.append(v);
        if (v >= 10) return sb.append('0').append(v);
        return sb.append("00").append(v);
    }
    private static StringBuilder append4(StringBuilder sb, int v) {
        if (v >= 1000) return sb.append(v);
        if (v >= 100) return sb.append('0').append(v);
        if (v >= 10) return sb.append("00").append(v);
        return sb.append("000").append(v);
    }

    private static void printSingleLine(String s) {
        if (USE_ANSI_CLEAR) {
            System.out.print(CR_CLEAR);
            System.out.print(s);
        } else {
            String padded = padRight(s, LINE_WIDTH);
            System.out.print("\r");
            System.out.print(padded);
        }
        System.out.flush();
    }

    private static String padRight(String s, int width) {
        if (s.length() >= width) return s.substring(0, width - 1);
        StringBuilder sb = new StringBuilder(width);
        sb.append(s);
        while (sb.length() < width) sb.append(' ');
        return sb.toString();
    }

    private static String truncate(String s, int maxWidth) {
        if (s.length() <= maxWidth) return s;
        if (maxWidth <= 3) return s.substring(0, Math.max(0, maxWidth));
        return s.substring(0, maxWidth - 3) + "...";
    }

    private static void sleepMillis(long ms) {
        try { Thread.sleep(ms); } catch (InterruptedException ignored) {}
    }

    private static Result queryNtp(String server, int timeoutMillis) throws Exception {
        byte[] req = new byte[48];
        req[0] = NTP_MODE_CLIENT_V4;

        InetAddress addr = InetAddress.getByName(server);
        DatagramPacket send = new DatagramPacket(req, req.length, addr, NTP_PORT);

        byte[] recvBuf = new byte[512];
        DatagramPacket recv = new DatagramPacket(recvBuf, recvBuf.length);

        try (DatagramSocket sock = new DatagramSocket()) {
            sock.setSoTimeout(timeoutMillis);

            long t1_ms = System.currentTimeMillis();
            sock.send(send);
            sock.receive(recv);
            long t4_ms = System.currentTimeMillis();

            int len = recv.getLength();
            if (len < 48) throw new IllegalStateException("NTP响应长度不足: " + len);
            byte[] data = recv.getData();

            double t2_unix = ntpTimestampToUnixSeconds(data, 32);
            double t3_unix = ntpTimestampToUnixSeconds(data, 40);
            double t1_unix = t1_ms / 1000.0;
            double t4_unix = t4_ms / 1000.0;

            double rtt = (t4_unix - t1_unix) - (t3_unix - t2_unix); // 秒
            double offset = ((t2_unix - t1_unix) + (t3_unix - t4_unix)) / 2.0; // 秒

            Result r = new Result();
            r.offsetSeconds = offset;
            r.rttMillis = Math.max(0, rtt * 1000.0);
            return r;
        }
    }

    private static double ntpTimestampToUnixSeconds(byte[] data, int offset) {
        long seconds = readUnsignedIntBE(data, offset);
        long fraction = readUnsignedIntBE(data, offset + 4);
        return (seconds - NTP_TIMESTAMP_DELTA) + (fraction / 4294967296.0);
    }

    private static long readUnsignedIntBE(byte[] data, int offset) {
        ByteBuffer bb = ByteBuffer.wrap(data, offset, 4).order(ByteOrder.BIG_ENDIAN);
        return bb.getInt() & 0xFFFFFFFFL;
    }

    private static class Result {
        double offsetSeconds; // 秒
        double rttMillis;     // 毫秒
    }

    // 简单容器
    private static final class YMDHMS {
        int year, month, day, hour, min, sec, millis;
    }
}