package distopt.utils


// spark-core
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd._
// spark-sql
import org.apache.spark.sql.SparkSession
// mllib
import org.apache.spark.mllib.linalg.{Vector, Vectors, Matrix, Matrices}
// others
import scala.math


object Utils {
    def loadLibsvmData(spark: SparkSession, filename: String, numSplits: Int): RDD[(Float, Array[Double])] = {
        // Loads data
        val rawdata = spark.read.format("libsvm")
                                .load(filename)
                                .rdd
                                .coalesce(numSplits)
        // note: coalesce can result in data being sent over the network. avoid this for large datasets
        
        val labelVectorRdd: RDD[(Float, Array[Double])] = rawdata
                .map(pair => (pair(0).toString.toFloat, Vectors.parse(pair(1).toString).toArray))
                .persist()
        
        labelVectorRdd
    }
    
    def normalize(sc: SparkContext, labelVectorRdd: RDD[(Float, Array[Double])]): RDD[(Double, Array[Double])] = {
        val maxFeatures: Array[Double] = labelVectorRdd.map(pair => pair._2.map(math.abs))
                                .reduce((a, b) => (a zip b).map(pair => math.max(pair._1, pair._2)) )
        val meanLabel: Double = labelVectorRdd.map(pair => pair._1)
                                .mean
        
        val maxFeaturesBc = sc.broadcast(maxFeatures)
        val meanLabelBc = sc.broadcast(meanLabel)
        
        val normalizedLabelVectorRdd: RDD[(Double, Array[Double])] = labelVectorRdd
            .map(pair => (pair._1-meanLabelBc.value, (pair._2 zip maxFeaturesBc.value).map(a => a._1 / a._2)))
        
        println(maxFeatures.mkString(","))
        println(meanLabel)
        
        normalizedLabelVectorRdd
    }
    
}