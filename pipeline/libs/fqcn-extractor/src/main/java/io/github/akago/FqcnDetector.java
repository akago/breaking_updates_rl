/*
 * Based on: CHAINS research project at KTH Royal Institute of Technology:bumper – FaultDetector.java
 * Source: https://github.com/chains-project/bumper
 * License: MIT
 *
 */

package io.github.akago;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import spoon.Launcher;
import spoon.reflect.CtModel;
import spoon.reflect.declaration.CtClass;
import spoon.reflect.declaration.CtElement;
import spoon.reflect.declaration.CtImport;
import spoon.reflect.declaration.CtMethod;
import spoon.reflect.declaration.CtType;
import spoon.reflect.visitor.filter.TypeFilter;
import spoon.support.reflect.declaration.CtFieldImpl;
import spoon.support.reflect.declaration.CtMethodImpl;
import spoon.support.reflect.declaration.CtParameterImpl;

public class FqcnDetector {
    private Set<MavenErrorLog.ErrorInfo> mavenErrorLog;

    public FqcnDetector(Set<MavenErrorLog.ErrorInfo> mavenErrorLog) {
        this.mavenErrorLog = mavenErrorLog;
    }

    public Set<String> detectFqcns(String projectFilePath) {
        Launcher spoon = new Launcher();
        spoon.getEnvironment().setAutoImports(true);
        spoon.addInputResource(projectFilePath);
        spoon.buildModel();

        CtModel model = spoon.getModel();
        Set<String> result = new HashSet<>();

        // Order is very important as you tipically want to fix from the first error
        result.addAll(getImportFaults(model));
        result.addAll(getFieldFaults(model));
        result.addAll(getMethodFaults(model));

        return result;
    }

    private MavenErrorLog.ErrorInfo getMavenErrorLog(CtElement element) {
        int startLineNumber = this.getRealLinePosition(element);
        int endLineNumber = element.getPosition().getEndLine();

        return mavenErrorLog
                    .stream()
                    .filter(mavenErrorLog -> {
                        int errorLineNumber = Integer.parseInt(mavenErrorLog.getClientLinePosition());
                        element.toString();
                        return errorLineNumber >= startLineNumber && errorLineNumber <= endLineNumber;
                    })
                    .findFirst()
                    .orElse(null);
    }

    private boolean containsAnError(CtElement element) {
        int startLineNumber = this.getRealLinePosition(element);
        int endLineNumber = element.getPosition().getEndLine();

        return mavenErrorLog.stream().anyMatch(mavenErrorLog -> {
            int errorLineNumber = Integer.parseInt(mavenErrorLog.getClientLinePosition());
            return errorLineNumber >= startLineNumber && errorLineNumber <= endLineNumber;
        });
    }

    private int getRealLinePosition(CtElement element) {
        // Need to do this trick as getLine does not take into account for decorators, and comments
        String[] lines = element.getOriginalSourceFragment().getSourceCode().split("\r\n|\r|\n");
        int numberOfLines = lines.length;
        return element.getPosition().getEndLine() - numberOfLines + 1;
    }

    private Set<String> getImportFaults(CtModel model) {
        CtType<?> mainClass = model.getAllTypes().iterator().next();
        Set<String> result = new HashSet<>();

        mainClass.getPosition().getCompilationUnit().getImports().stream()
            .forEach((CtElement element) -> {
                if(this.containsAnError(element)) {
                    Set<String> fqcns = FqcnUtils.collectFqcns(element, true, false, false);
                    result.addAll(fqcns);
                }
            });

        return result;
    }

    private Set<String> getFieldFaults(CtModel model) {
        CtType<?> mainClass = model.getAllTypes().iterator().next();
        Set<String> result = new HashSet<>();

        mainClass.getElements(new TypeFilter<>(CtFieldImpl.class)).stream()
            .forEach((CtElement element) -> {
                if(this.containsAnError(element)) {
                    Set<String> fqcns = FqcnUtils.collectFqcns(element, true, false, false);
                    result.addAll(fqcns);
                }
            });

        return result;
    }

    private Set<String> getMethodFaults(CtModel model) {
        CtType<?> mainClass = model.getAllTypes().iterator().next();
        Set<String> result = new HashSet<>();

        mainClass.getElements(new TypeFilter<>(CtMethodImpl.class)).stream().forEach(element -> {
            if(this.containsAnError(element)) {
                Set<String> fqcns = FqcnUtils.collectFqcns(element, true, false, false);
                result.addAll(fqcns);
            }
        });
        return result;
    }
}
