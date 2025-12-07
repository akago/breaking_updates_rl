package dev.yourorg.tools;

import dev.yourorg.tools.japicmp_analyzer.ApiChange;
import dev.yourorg.tools.japicmp_analyzer.ApiChangeType;
import dev.yourorg.tools.japicmp_analyzer.ApiMetadata;
import dev.yourorg.tools.japicmp_analyzer.JApiCmpAnalyze;
import japicmp.model.JApiClass;

import java.nio.file.Path;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

public class AdditionsOnlyMain {
    public static void main(String[] args) {
        if (args.length != 2) {
            System.err.println("Usage: java dev.yourorg.tools.AdditionsOnlyMain <old-jar> <new-jar>");
            System.exit(1);
        }

        Path oldJar = Path.of(args[0]).toAbsolutePath().normalize();
        Path newJar = Path.of(args[1]).toAbsolutePath().normalize();

        ApiMetadata oldApi = new ApiMetadata(oldJar.getFileName().toString(), oldJar);
        ApiMetadata newApi = new ApiMetadata(newJar.getFileName().toString(), newJar);

        JApiCmpAnalyze analyzer = new JApiCmpAnalyze(oldApi, newApi);
        List<JApiClass> rawChanges = analyzer.getChanges();
        Set<ApiChange> apiChanges = analyzer.getAllChanges(rawChanges);

        List<ApiChange> additions = apiChanges.stream()
                .filter(change -> ApiChangeType.ADD.equals(change.getAction()))
                .sorted((a, b) -> a.getLongName().compareToIgnoreCase(b.getLongName()))
                .collect(Collectors.toList());

        if (additions.isEmpty()) {
            System.out.println("No API additions found.");
            return;
        }

        additions.forEach(change -> {
            System.out.println(change.toDiffString());
            // 若想要 JSON，可改成 System.out.println(change.toString());
        });
    }
}
