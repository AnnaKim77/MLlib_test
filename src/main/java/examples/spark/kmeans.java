package examples.spark;

import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.*;
import udf.VectorBuilder;

import static org.apache.spark.sql.functions.callUDF;

/**
 * Created by ubuntu on 16. 8. 22.
 */
public class kmeans {

    public static void main(String[] args) {

        SparkSession spark = SparkSession.builder().appName("K-means")
                .master("local[*]").getOrCreate();

        //Load data

        spark.udf().register("vectorBuilder", new VectorBuilder(), new VectorUDT());
        String filename = "src/main/resources/iris.txt";
        StructType schema = new StructType(
                new StructField[]{
                        new StructField("f1", DataTypes.DoubleType, false, Metadata.empty()),
                        new StructField("f2", DataTypes.DoubleType, false, Metadata.empty()),
                        new StructField("f3", DataTypes.DoubleType, false, Metadata.empty()),
                        new StructField("f4", DataTypes.DoubleType, false, Metadata.empty()),
                        new StructField("label", DataTypes.DoubleType, false, Metadata.empty()),
                        new StructField("feature", new VectorUDT(), true, Metadata.empty())
                });
        Dataset<Row> df = spark.read().format("csv").schema(schema).option("header", "false").load(filename);
        df = df.withColumn("features",
                callUDF("vectorBuilder", df.col("f1"), df.col("f2"), df.col("f3"), df.col("f4")));

        // Train k-means model (vector type만 feature로 인식함)
        KMeans kmeans = new KMeans().setK(3);
        KMeansModel model = kmeans.fit(df);

        double SSE = model.computeCost(df);

        Vector[] centers = model.clusterCenters();
        System.out.println("cluster centers :");
        for (Vector center : centers) {
            System.out.println(center);
        }

        spark.stop();
    }

}
