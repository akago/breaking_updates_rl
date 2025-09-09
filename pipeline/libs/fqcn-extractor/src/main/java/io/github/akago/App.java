package io.github.akago;

import spoon.Launcher;
import spoon.reflect.CtModel;
import java.util.Set;
import java.util.HashSet;
import java.util.Map;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Path;

import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectWriter;


public class App {
    public static void main(String[] args) {
        int exitCode = new CommandLine(new FqcnExtractor()).execute(args);
        System.exit(exitCode);
    }

    @CommandLine.Command(name = "FqcnExtractor", mixinStandardHelpOptions = true, description = "Extracts relevant FQCNs from a Java project and a build log file.")
    private static class FqcnExtractor implements Runnable {

        @CommandLine.Option(
                names = {"-c", "--client"},
                paramLabel = "Client project",
                description = "A client project to analyze.",
                required = true
        )
        Path client;

        @CommandLine.Option(
                names = {"-l", "--log"},
                paramLabel = "Maven log",
                description = "The maven log to analyze.",
                required = false
        )
        File mavenLog;

    @Override
    public void run() {
        
        MavenLogAnalyzer mavenLog = new MavenLogAnalyzer(this.mavenLog);
        Set<String> fqcns = new HashSet<>();
        String project = this.client.toString();

        try {
            MavenErrorLog log = mavenLog.analyzeCompilationErrors();
            // collect FQCNs
            log.getErrorInfo().forEach((k, v) -> {
                // create detector for each file name
                FqcnDetector detector = new FqcnDetector(v);
                Set<String> result = detector.detectFqcns(project + k);
                fqcns.addAll(result);
            });

            // ObjectWriter ow = new ObjectMapper().writer().withDefaultPrettyPrinter();
            new ObjectMapper().writerWithDefaultPrettyPrinter().writeValue(System.out, fqcns);

        }catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

}
}
