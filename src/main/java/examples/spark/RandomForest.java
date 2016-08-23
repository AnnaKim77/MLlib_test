package examples.spark;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.Dataset;
import udf.VectorBuilder;

import static org.apache.spark.sql.functions.callUDF;

/**
 * Created by ubuntu on 16. 8. 18.
 * author : Kwang-Hee Kim
 */
public class RandomForest {

    public static void main(String[] args) {

        // 스파크 환경 세팅
        SparkSession spark = SparkSession.builder().appName("RandomForest_R")
                .master("local[*]").getOrCreate();


        // Load data as data frame & split
        // Features : 모델에 사용되는 값(value)으로 vector 형태로 변환이 필요함.
        // Label : classification 에 사용되는 카테고리 변수 (ex. iris의 Species)
        spark.udf().register("vectorBuilder", new VectorBuilder(), new VectorUDT());
        String filename = "src/main/resources/iris.txt";
        StructType schema = new StructType(
                new StructField[]{new StructField("f1", DataTypes.DoubleType, false, Metadata.empty()),
                        new StructField("f2", DataTypes.DoubleType, false, Metadata.empty()),
                        new StructField("f3", DataTypes.DoubleType, false, Metadata.empty()),
                        new StructField("f4", DataTypes.DoubleType, false, Metadata.empty()),
                        new StructField("label", DataTypes.DoubleType, false, Metadata.empty()),
                        new StructField("features", new VectorUDT(), true, Metadata.empty())});
        Dataset<Row> df = spark.read().format("csv").schema(schema).option("header", "false").load(filename);

        df = df.withColumn("features",
                callUDF("vectorBuilder", df.col("f1"), df.col("f2"), df.col("f3"), df.col("f4")));
        df.printSchema();
        df.show();

        Dataset<Row>[] splits = df.randomSplit(new double[]{0.8, 0.2});
        Dataset<Row> trainSet = splits[0];
        Dataset<Row> testSet = splits[1];


        // Index labels, adding metadata to the label column.
        // Fit on whole data-set to include all labels in index.
        // Automatically identify categorical features, and index them.
        StringIndexerModel labelIndexer = new StringIndexer()
                .setInputCol("label")
                .setOutputCol("indexedLabel")
                .fit(df);
        VectorIndexerModel featureIndexer = new VectorIndexer()
                .setInputCol("features")
                .setOutputCol("indexedFeatures")
                .fit(df);
        IndexToString labelConverter = new IndexToString()
                .setInputCol("prediction")
                .setOutputCol("predictedLabel")
                .setLabels(labelIndexer.labels());


        // Train a Random Forest model
        RandomForestClassifier rf = new RandomForestClassifier()
                .setLabelCol("indexedLabel")
                .setFeaturesCol("indexedFeatures");


        // Pipeline 역할: Estimator(Model fit) , Transformer(produce the dataset for the next stage)
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{labelIndexer, featureIndexer, rf, labelConverter});
        PipelineModel model = pipeline.fit(trainSet);

        // Prediction
        Dataset<Row> predictions = model.transform(testSet);
        predictions.select("predictedLabel", "label", "features").show(10);
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("indexedLabel")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        double accuracy = evaluator.evaluate((predictions));
        System.out.println("Test Error =" + (1.0 - accuracy));
    }

}
