package examples.etc;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;


public class RandomForestExample {
    public static void main(String[] args) {

        // Spark env. setting
        SparkSession spark = SparkSession
                .builder().master("local[*]")
                .appName("RandomForest")
                .getOrCreate();
        // Load the data stored in LIBSVM format as a DataFrame.
        Dataset<Row> data = spark
                .read()
                .format("libsvm")
                .load("src/main/resources/libsvm_iris.txt");
        System.out.println("dataset : ");

        data.show();

        // Index labels, adding metadata to the label column.
        // Fit on whole dataset to include all labels in index.
        StringIndexerModel labelIndexer = new StringIndexer()
                .setInputCol("label")
                .setOutputCol("indexedLabel")
                .fit(data);

        // Automatically identify categorical features, and index them.
        VectorIndexerModel featureIndexer = new VectorIndexer()
                .setInputCol("features")
                .setOutputCol("indexedFeatures")
                .setMaxCategories(3)
                .fit(data);

        System.out.println("After indexed :");
        data.show();

        // Split the data into training and test sets (30% held out for testing).
        Dataset<Row>[] splits = data.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

        // Train a RandomForest model
        RandomForestClassifier rf = new RandomForestClassifier()
                .setLabelCol("indexedLabel")
                .setFeaturesCol("indexedFeatures");

        // Convert indexed labels back to original labels.
        IndexToString labelConverter = new IndexToString()
                .setInputCol("prediction")
                .setOutputCol("predictedLabel")
                .setLabels(labelIndexer.labels());

        // Chain indexers and tree in a Pipeline.
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{labelIndexer, featureIndexer, rf, labelConverter});

        // Train model. This also runs the indexers.
        PipelineModel model = pipeline.fit(trainingData);

        // Make predictions.
        Dataset<Row> predictions = model.transform(testData);

        // Select example rows to display.
        predictions.select("predictedLabel", "label", "features").show(5);

        // Select (prediction, true label) and compute test error.
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("indexedLabel")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Test Error = " + (1.0 - accuracy));

        RandomForestClassificationModel rfModel = (RandomForestClassificationModel) (model.stages()[2]);
        System.out.println("Learned classification forest model :\n" + rfModel.toDebugString());
    }

}
