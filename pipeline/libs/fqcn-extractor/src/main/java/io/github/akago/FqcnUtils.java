package io.github.akago;

import spoon.reflect.declaration.*;
import spoon.reflect.code.*;
import spoon.reflect.reference.*;
import spoon.reflect.visitor.filter.TypeFilter;
import spoon.reflect.CtModel;

import java.util.*;
import java.util.function.Predicate;

import java.util.regex.Pattern;
import java.util.regex.Matcher;

public class FqcnUtils {

  
    private static final Pattern IMPORT_PATTERN = Pattern.compile(
        "^import\\s+(?:static\\s+)?([\\w\\.]+)(?:\\s*\\.\\*)?\\s*;?$"
    );


    /*
        * Collect FQCNs under a CtElement (could be a field or method ).
        *
        * @param root
        * @param withMembers if true, also add "Type.member" (method/field simple names)
        * @param withSignatures if true, also add "Type#method(paramTypes...)" signatures
    */
    public static Set<String> collectFqcns(CtElement root, boolean withMembers, boolean withSignatures) {
        Set<String> out = new LinkedHashSet<>();
        
        // find all references under the root element
        for (CtReference ref : root.getElements(new TypeFilter<>(CtReference.class))) {
            handleSingleReference(out, ref, withMembers, withSignatures);
        }

        // for (CtTypeAccess<?> ta : root.getElements(new TypeFilter<>(CtTypeAccess.class))) {
        //     CtTypeReference<?> tr = ta.getAccessedType();
        //     addType(out, tr);
        // }

        for (CtAnnotation an : root.getElements(new TypeFilter<>(CtAnnotation.class))) {
            CtTypeReference<?> tr = an.getAnnotationType();
            addType(out, tr);
        }

        return out;

    }

    /*
        * Extract FQCN from an CtImport statement.
        * Return null if the import statement is invalid
        *
        * @param imp
        * @return the FQCN string, or null if not found
    */
    public static String extractFromImport(CtElement imp) {
        if (imp == null) return null;
        String s = imp.toString().trim();
        Matcher m = IMPORT_PATTERN.matcher(s);
        if (!m.matches()) return null;
        return m.group(1);
    }

    private static void handleSingleReference(Set<String> out,
                                              CtReference ref,
                                              boolean withMembers,
                                              boolean withSignatures) {
        if (ref instanceof CtTypeReference<?> tr) {
            // System.out.println("Type: " + tr);
            addType(out, tr);
        } else if (ref instanceof CtExecutableReference<?> er) {
            // System.out.println("Method: " + er);
            CtTypeReference<?> decl = er.getDeclaringType();
            if (decl == null) return; // could be null for local methods
            String typeQN = qn(decl);
            if (typeQN != null) {
                out.add(typeQN);
                if (withMembers)    out.add(typeQN + "." + er.getSimpleName());
                if (withSignatures) out.add(typeQN + "#" + er.getSignature());
            }

        } else if (ref instanceof CtFieldReference<?> fr) {
            CtTypeReference<?> decl = fr.getDeclaringType();
            if (decl == null) return;
            String typeQN = qn(decl);
            if (typeQN != null) {
                out.add(typeQN);
                if (withMembers) 
                    out.add(typeQN + "." + fr.getSimpleName());
            }
        } 
    }

    // ---- Helpers ----
    private static void addType(Set<String> out, CtTypeReference<?> tr) {
        if (tr == null) return;
        if (tr.isPrimitive()) return;
        if (tr instanceof CtTypeParameterReference) return; // Skip Type Parameters like T, E
        if (tr instanceof CtWildcardReference) return;       // Skip Wildcard annotation '?'

        // Array type
        if (tr instanceof CtArrayTypeReference<?> arr) {
            addType(out, arr.getComponentType());
            return;
        }

        String qn = qn(tr);
        if (qn != null) out.add(qn);

        // Generic type arguments
        List<CtTypeReference<?>> actuals = tr.getActualTypeArguments();
        if (actuals != null) {
            for (CtTypeReference<?> a : actuals) addType(out, a);
        }

        // Inner class -> Add outer class
        CtTypeReference<?> outer = tr.getDeclaringType();
            if (outer != null) addType(out, outer);
    }

    private static String qn(CtTypeReference<?> tr) {
        if (tr == null) return null;
        String qn = tr.getQualifiedName();
        if (qn == null || qn.isEmpty() || "?".equals(qn)) return null;
        return qn;
    }

}
