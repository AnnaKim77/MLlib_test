package examples

import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg._
import org.apache.spark.sql.SparkSession


/**
  * Created by anna on 17. 3. 22.
  */
object AnomalyDetection {
  def main(args: Array[String]): Unit = {

    val origfile = "src/main/resources/kddcup.data.corrected"
    val jsonFields = List("duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label")
    val jsonFile = origfile + ".json"
    val json = scala.io.Source.fromFile(origfile).getLines.map { line =>
      jsonFields.zip(line.split(","))
        .toMap
        .map { case (k, v) => s""" "$k": "$v" """ }
        .mkString("{", ",", "}")
    }
    val w = new java.io.FileWriter(new java.io.File(jsonFile))
    json.foreach { j => w.write(j + "\n") }
    w.close

    val spark = SparkSession
      .builder().master("local")
      .config("spark.some.config.option", "some-value")
      .getOrCreate()

    // For implicit conversions like converting RDDs to DataFrames
    import spark.implicits._


    val dataFrame = spark.read.json(jsonFile).cache

    dataFrame.printSchema
    dataFrame.count


    val labelsCount = dataFrame.groupBy("label").count().collect
    labelsCount.toList.map(row => (row.getString(0), row.getLong(1)))


    val nonNumericFrame = List("protocol_type", "service", "flag")
    val labeledNumericFrame = dataFrame.select(
      "label",
      "duration",
      "src_bytes",
      "dst_bytes",
      "land",
      "wrong_fragment",
      "urgent",
      "hot",
      "num_failed_logins",
      "logged_in",
      "num_compromised",
      "root_shell",
      "su_attempted",
      "num_root",
      "num_file_creations",
      "num_shells",
      "num_access_files",
      "num_outbound_cmds",
      "is_host_login",
      "is_guest_login",
      "count",
      "srv_count",
      "serror_rate",
      "srv_serror_rate",
      "rerror_rate",
      "srv_rerror_rate",
      "same_srv_rate",
      "diff_srv_rate",
      "srv_diff_host_rate",
      "dst_host_count",
      "dst_host_srv_count",
      "dst_host_same_srv_rate",
      "dst_host_diff_srv_rate",
      "dst_host_same_src_port_rate",
      "dst_host_srv_diff_host_rate",
      "dst_host_serror_rate",
      "dst_host_srv_serror_rate",
      "dst_host_rerror_rate",
      "dst_host_srv_rerror_rate"
    )

    labeledNumericFrame.take(1)(0)

    val labeledPoint = labeledNumericFrame.map(row =>
      (row.getString(0), Vectors.dense(row.toSeq.tail.map(s => if (s == null) 0.0 else s.toString.toDouble).toArray)))

    val rawData = labeledPoint.rdd.values
    rawData.first


    val scaler = new StandardScaler().fit(rawData)
    val data = scaler.transform(rawData).cache
    data.first.toArray.toList


    //k-means clustering
    val numIterations = 10
    //in production it should be more
    val K = 150
    val clusters = KMeans.train(data, K, numIterations)


    @transient val ser = new java.io.Serializable {
      val lp = labeledPoint
      val cs = clusters
      val sc = scaler
      val f = (x: (String, org.apache.spark.mllib.linalg.Vector)) => (cs.predict(sc.transform(x._2)), x._1)
      val predictions = lp.map(x => f(x))
    }
    ser.predictions

    val clustersWithSize = ser.predictions.rdd.map(x => (x._1, 1)).reduceByKey((x, y) => x + y)
    clustersWithSize.take(25).toList


    val clustersWithCountAndLabel = clustersWithSize.join(ser.predictions.rdd).distinct
    clustersWithCountAndLabel.take(20).toList

    //clusters with 1 point and labeled as normal
    val suspectedAnomalousClusters = clustersWithCountAndLabel.filter(x => x._2._1 == 1 && x._2._2 == "normal.").map(x => x._1)

    val anomalousClusters = suspectedAnomalousClusters.collect
    anomalousClusters.toList


  }

}
