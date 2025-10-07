import spoon.Launcher;
import spoon.reflect.CtModel;
import spoon.reflect.declaration.CtCompilationUnit;
import spoon.support.compiler.FileSystemFile;

import java.io.File;

public class RemoveCommentsStdout {
    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            System.err.println("Usage: java -jar remove-comments.jar <JavaSourceFile>");
            System.exit(1);
        }

        String filePath = args[0];
        File inputFile = new File(filePath);

        if (!inputFile.exists()) {
            System.err.println("File not found: " + filePath);
            System.exit(1);
        }


        Launcher launcher = new Launcher();
        launcher.addInputResource(new FileSystemFile(inputFile));
        launcher.getEnvironment().setCommentEnabled(false); 
        launcher.buildModel();

        CtModel model = launcher.getModel();


        for (CtCompilationUnit cu : launcher.getFactory().CompilationUnit().getMap().values()) {
            System.out.println(cu.prettyprint());
        }
    }
}