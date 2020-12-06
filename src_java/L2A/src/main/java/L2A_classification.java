import weka.core.Instances;
import weka.core.SelectedTag;

import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.output.prediction.PlainText;

import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.Namespace;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import static net.sourceforge.argparse4j.impl.Arguments.storeTrue;

import java.io.File;
import java.io.FileWriter;
import java.io.FileReader;
import java.io.BufferedWriter;
import java.io.BufferedReader;


import java.util.Random;
import java.io.IOException;


public class L2A_classification {

	static Integer n_fold = 5;

	public static void main(String[] args) throws Exception {
		
		String path_input = "";
		String path_output = "";

		ArgumentParser parser = ArgumentParsers.newFor("Checksum").build().defaultHelp(true)
				.description("Calculate checksum of given files.");
		
		parser.addArgument("-path_input", "--i").help("path of input file").action(storeTrue());
		parser.addArgument("-path_output", "--o").help("path of output file").action(storeTrue());
		
		Namespace res;
		try {
            res = parser.parseArgs(args);
            System.out.println(res);
        } catch (ArgumentParserException e) {
            parser.handleError(e);
            System.exit(1);
        }
		

		// 1. INITIALIZE CLASSIFIERS

		Classifier[] classifiers = new Classifier[] {};
		String[] classifierNames = new String[] {};

		// Future work: implement choice for each classifier
		LibSVM linsvm = new LibSVM();
		linsvm.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
		classifiers = new Classifier[] { linsvm };
		classifierNames = new String[] { "SVM-linear" };

		for (int i = 0; i < classifiers.length; i++) {

			// 2. SET OUTPUT FOLDER
			
			path_output = path_output + "/performance.csv";

			String path_prediction = path_output + "/prediction.csv";

			File file = new File(path_output);
			if (!file.exists()) {
				file.createNewFile();
			}
			BufferedWriter bw = new BufferedWriter(new FileWriter(file.getAbsoluteFile(), false));
			String header = "ML Model" + ";" + "Representation Model" + ";" + "Classifier" + ";" + "Accuracy" + ";"
					+ "Precision" + ";" + "Recall" + ";" + "F-measure" + "\n";
			bw.write(header);
			bw.flush();

			// 3. RUN CLASSIFICATION

			Classifier c = classifiers[i];
			String c_name = classifierNames[i];

			System.out.println("PATH INPUT " + path_input);
			System.out.println("CLASSIFIER " + c_name);
			System.out.println("SAVING RESULTS " + path_output);

			classification(path_input, path_prediction, c, c_name, bw);

			bw.flush();

			bw.close();

		}

	}

	public static void classification(String path_arff, String path_prediction, Classifier c, String c_name,
			BufferedWriter bw) throws Exception {

		System.out.println("Read Data");

		Instances train = readData(path_arff);
		train.setClassIndex(train.numAttributes() - 1);

		FilteredClassifier fc = new FilteredClassifier();
		fc.setClassifier(c);

		classifyCV(train, path_prediction, n_fold, fc, c_name, bw, false);

	}

	public static void classifyCV(Instances data, String path_prediction, int folds, Classifier cls, String c_name,
			BufferedWriter bw, boolean more_details) throws Exception {

		Evaluation eval = new Evaluation(data);
		StringBuffer predsBuffer = new StringBuffer();
		PlainText plainText = new PlainText();
		plainText.setHeader(data);
		plainText.setBuffer(predsBuffer);
		Random rand = new Random(1); // using seed = 1
		eval.crossValidateModel(cls, data, folds, rand, plainText);

		File file = new File(path_prediction);

		if (!file.exists()) {
			file.createNewFile();
		}
		BufferedWriter bw_pred = new BufferedWriter(new FileWriter(file.getAbsoluteFile(), false));
		bw_pred.write(predsBuffer.toString());
		bw_pred.flush();
		bw_pred.close();

		if (more_details) {
			System.out.println();
			System.out.println("=== Setup ===");
			System.out.println("Classifier: " + cls.getClass().getName());
			System.out.println("Dataset: " + data.relationName());
			System.out.println("Folds: " + folds);
			// System.out.println("Seed: " + seed);
			System.out.println();
			System.out.println(eval.toSummaryString("=== " + folds + "-fold Cross-validation ===", false));
			System.out.println(eval.toClassDetailsString());
			System.out.println(eval.toMatrixString());

		}

		double acc = eval.pctCorrect() / 100;
		String accuracy = Double.toString(acc).replace(".", ",");

		String fmeasure = Double.toString(eval.weightedFMeasure()).replace(".", ",");
		String precision = Double.toString(eval.weightedPrecision()).replace(".", ",");
		String recall = Double.toString(eval.weightedRecall()).replace(".", ",");

		String results = c_name + ";" + accuracy + ";" + precision + ";" + recall + ";" + fmeasure + "\n";
		bw.write(results);

	}

	public static Instances readData(String path) {
		System.out.println(path.toUpperCase());
		Instances data = null;
		try {
			BufferedReader reader = new BufferedReader(new FileReader(path));
			data = new Instances(reader);
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

		return data;
	}

}
