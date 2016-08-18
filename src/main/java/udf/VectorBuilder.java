package udf;

import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.api.java.UDF4;

/**
 * Created by ubuntu on 16. 8. 17.
 */

public class VectorBuilder implements UDF4 <Double, Double,Double,Double,Vector> {
    private static final long serialVersionUID = -2991355883253063841L;

    @Override
    public Vector call(Double t1,Double t2,Double t3,Double t4 ) throws Exception {
        return Vectors.dense(t1,t2,t3,t4);

    }

}
