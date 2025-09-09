package io.github.akago;

import spoon.reflect.declaration.*;
import spoon.reflect.code.*;
import spoon.reflect.reference.*;
import spoon.reflect.visitor.filter.TypeFilter;
import spoon.reflect.CtModel;

import java.util.*;
import java.util.function.Predicate;

public class FqcnUtils {

  /**
   * Collect FQCNs under a CtElement (could be a field, method or an import ).
   *
   * @param root            
   * @param withMembers     if true, also add "Type.member" (method/field simple names)
   * @param withSignatures  if true, also add "Type#method(paramTypes...)" signatures
   * @param excludeJdkPkgs  if true, drop java.*, javax.*, jdk.*, sun.*, com.sun.*
   */
  public static Set<String> collectFqcns(CtElement root,
                                         boolean withMembers,
                                         boolean withSignatures,
                                         boolean excludeJdkPkgs) {
    Set<String> out = new LinkedHashSet<>();

    // Type references 
    for (CtTypeReference<?> tr : root.getElements(new TypeFilter<>(CtTypeReference.class))) {
      addType(out, tr);
    }

    // Executable references (methods/constructors)
    for (CtExecutableReference<?> execRef : root.getElements(new TypeFilter<>(CtExecutableReference.class))) {
      CtTypeReference<?> decl = execRef.getDeclaringType();
      if (decl != null) {
        String typeQN = qn(decl);
        if (typeQN != null) {
          out.add(typeQN); // declaring type FQCN
          if (withMembers)    out.add(typeQN + "." + execRef.getSimpleName());
          if (withSignatures) out.add(typeQN + "#" + execRef.getSignature());
        }
      }
    }

    // Field references (include static fields)
    for (CtFieldReference<?> fieldRef : root.getElements(new TypeFilter<>(CtFieldReference.class))) {
      CtTypeReference<?> decl = fieldRef.getDeclaringType();
      if (decl != null) {
        String typeQN = qn(decl);
        if (typeQN != null) {
          out.add(typeQN);
          if (withMembers) out.add(typeQN + "." + fieldRef.getSimpleName());
        }
      }
    }

    return out;
  }

  // ---- Helpers -------------------------------------------------------------

  /* Add a type ref and everything "hidden inside" (arrays, generics, outer types)*/
  private static void addType(Set<String> out, CtTypeReference<?> tr) {
    if (tr == null) return;

    // Skip primitives (int, boolean, …)
    if (tr.isPrimitive()) return;

    // Arrays
    if (tr instanceof CtArrayTypeReference) {
      CtTypeReference<?> comp = ((CtArrayTypeReference<?>) tr).getComponentType();
      if (comp != null) addType(out, comp);
      return;
    }

    // Record the qualified name if available.
    String qn = qn(tr);
    if (qn != null) out.add(qn);

    // Expand generic actual type arguments
    List actuals = tr.getActualTypeArguments();
    if (actuals != null) {
      for (Object o : actuals) {
        if (o instanceof CtTypeReference) {
          addType(out, (CtTypeReference<?>) o);
        }
      }
    }

    // For inner classes, also include the declaring (outer) type if present.
    CtTypeReference<?> outer = tr.getDeclaringType();
    if (outer != null) addType(out, outer);
  }

  /* Normalize Spoon's qualified name; return null for unresolved/placeholder names. */
  private static String qn(CtTypeReference<?> tr) {
    String qn = tr.getQualifiedName();
    return (qn == null || qn.isEmpty() || "?".equals(qn)) ? null : qn;
  }


  /* Human-readable label for grouping (class#method(signature) or class.field). */
  public static String label(CtElement e) {
    if (e instanceof CtMethod<?> m) {
      String owner = m.getDeclaringType() != null ? m.getDeclaringType().getQualifiedName() : "<unknown>";
      return owner + "#" + m.getSignature();
    }
    if (e instanceof CtField<?> f) {
      String owner = f.getDeclaringType() != null ? f.getDeclaringType().getQualifiedName() : "<unknown>";
      return owner + "." + f.getSimpleName();
    }
    if (e instanceof CtType<?> t) {
      return t.getQualifiedName();
    }
    return e.getClass().getSimpleName();
  }

  /* Produce FQCN sets per field/method/import */
  public static Map<String, Set<String>> listFqcnsPerMember(List<CtElement> nodes,
                                                            boolean withMembers,
                                                            boolean withSignatures,
                                                            boolean excludeJdkPkgs) {

    Map<String, Set<String>> result = new LinkedHashMap<>();
    for (CtElement e : nodes) {
      String key = label(e);
      Set<String> fqcn = collectFqcns(e, withMembers, withSignatures, excludeJdkPkgs);
      result.put(key, fqcn);
    }
    return result;
  }

}
