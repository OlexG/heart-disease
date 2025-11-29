import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Minimal logger that mirrors console output while capturing lines for later persistence.
 */
public class ProcessLogger {
    private final List<String> entries = new ArrayList<>();

    public void info(String message) {
        log(message, false);
    }

    public void error(String message) {
        log(message, true);
    }

    public List<String> getEntries() {
        return Collections.unmodifiableList(entries);
    }

    private void log(String message, boolean isError) {
        if (isError) {
            System.err.println(message);
        } else {
            System.out.println(message);
        }
        entries.add(message);
    }
}

