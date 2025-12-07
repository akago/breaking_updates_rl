package dev.yourorg.tools.japicmp_analyzer;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;

public class ApiMetadata {
    private final String name;
    private final Path path;

    public ApiMetadata(String name, Path path) {
        this.name = name;
        this.path = path;
    }

    public String getName() {
        return name;
    }

    public Path getPath() {
        return path;
    }

    /**
     * List all classes from a jar file.
     *
     * @return List of class names.
     */
    public List<String> listAllClassFromJar() {
        try (JarFile jarFile = new JarFile(path.toFile())) {
            List<String> classNames = new ArrayList<>();
            Enumeration<JarEntry> entries = jarFile.entries();

            while (entries.hasMoreElements()) {
                JarEntry entry = entries.nextElement();
                if (entry.isDirectory() || !entry.getName().endsWith(".class")) {
                    continue;
                }
                // strip .class suffix and convert path separators.
                String className = entry.getName().substring(0, entry.getName().length() - 6);
                className = className.replace('/', '.');
                classNames.add(className);
            }

            return classNames;
        } catch (IOException | SecurityException e) {
            throw new RuntimeException(e);
        }
    }
}
