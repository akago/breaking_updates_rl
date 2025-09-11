package io.github.akago;

import spoon.reflect.declaration.*;
import spoon.reflect.code.*;
import spoon.reflect.reference.*;
import spoon.reflect.visitor.filter.TypeFilter;
import spoon.reflect.CtModel;

import java.util.*;
import java.util.function.Predicate;

public class FqcnUtils {

  /*
   * Collect FQCNs under a CtElement (could be a field, method or an import ).
   *
   * @param root
   * @param withMembers if true, also add "Type.member" (method/field simple names)
   * @param withSignatures if true, also add "Type#method(paramTypes...)" signatures
   */
    public static Set<String> collectFqcns(CtElement root, boolean withMembers, boolean withSignatures) {
        Set<String> out = new LinkedHashSet<>();
        
        // special case for CtImport
        if (root instanceof CtImport imp) {
            CtReference ref = imp.getReference();
            handleSingleReference(out, ref, withMembers, withSignatures);
            return out;
        }
        // find all references under the root element
        for (CtReference ref : root.getElements(new TypeFilter<>(CtReference.class))) {
            handleSingleReference(out, ref, withMembers, withSignatures);
        }
        return out;

    }

    private static void handleSingleReference(Set<String> out,
                                              CtReference ref,
                                              boolean withMembers,
                                              boolean withSignatures) {
        if (ref instanceof CtTypeReference<?> tr) {
            addType(out, tr);

        } else if (ref instanceof CtExecutableReference<?> er) {
            CtTypeReference<?> dt = er.getDeclaringType();
            String typeQN = qn(dt);
            if (typeQN != null) {
                out.add(typeQN);
                if (withMembers)    out.add(typeQN + "." + er.getSimpleName());
                if (withSignatures) out.add(typeQN + "#" + er.getSignature());
            }

        } else if (ref instanceof CtFieldReference<?> fr) {
            // import static a.b.C.*;
            if ("*".equals(fr.getSimpleName())) {
                CtTypeReference<?> dt = fr.getDeclaringType();
                addType(out, dt); // only record the type
                return;
            }
            CtTypeReference<?> dt = fr.getDeclaringType();
            String typeQN = qn(dt);
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
        String qn = tr.getQualifiedName();
        if (qn == null || qn.isEmpty() || "?".equals(qn)) return null;
        return qn;
    }

}
