package distopt

// spark-core
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd._
// spark-sql
import org.apache.spark.sql.SparkSession
// spark-mllib
//import org.apache.spark.mllib.linalg.{Vector, Vectors, Matrix, Matrices, DenseMatrix, DenseVector}
//import org.apache.spark.mllib.linalg.distributed.RowMatrix
//import org.apache.spark.mllib.linalg.SingularValueDecomposition
// breeze
import breeze.linalg._
import breeze.numerics._
// others
import scala.math
import java.io._


import distopt.utils._
import distopt.quadratic._

object Quadratic {
    def main(args: Array[String]) {
        // Parse parameters from command line arguments
        val filename: String = args(0).toString
        val numSplits: Int = args(1).toInt
        val gamma: Double = args(2).toDouble
        val maxiter: Int = args(3).toInt
        
        // Launch Spark
        var t0 = System.nanoTime()
        val spark = (SparkSession
                      .builder()
                      .appName("Giant for Ridge Regression")
                      .config("spark.some.config.option", "some-value")
                      .getOrCreate())
        val sc = spark.sparkContext
        //sc.setLogLevel("ERROR")
        var t1 = System.nanoTime()
        println("Time cost of starting Spark:  " + ((t1-t0)*1e-9).toString + "  seconds.")
        
        
        // Load data
        t0 = System.nanoTime()
        val dataRaw: RDD[(Float, Array[Double])] = Utils.loadLibsvmData(spark, filename, numSplits).persist()
        println("Number of samples: " + dataRaw.count.toString)
        println("Number of partitions: " + dataRaw.getNumPartitions.toString)
        t1 = System.nanoTime()
        println("Time cost of loading data:  " + ((t1-t0)*1e-9).toString + "  seconds.")
        
        // Normalize Data
        t0 = System.nanoTime()
        val (meanLabel, maxFeatures) = Utils.meanAndMax(dataRaw)
        val data: RDD[(Double, Array[Double])] = Utils.normalize(sc, dataRaw, meanLabel, maxFeatures).persist()
        val n: Long = data.count
        t1 = System.nanoTime()
        println("n = " + n.toString)
        println("Time cost of data normalization:  " + ((t1-t0)*1e-9).toString + "  seconds.")
        println(" ")        
        
        println("####################################")
        println("spark.conf.getAll:")
        spark.conf.getAll.foreach(println)
        println(" ")
        println("getExecutorMemoryStatus:")
        println(sc.getExecutorMemoryStatus.toString())
        println("####################################")
        println(" ")
        
        
        // Test data
        val dataTestRaw: RDD[(Float, Array[Double])] = Utils.loadLibsvmData(spark, filename+".t", numSplits)
        val dataTest: RDD[(Double, Array[Double])] = Utils.normalize(sc, dataTestRaw, meanLabel, maxFeatures).persist()
        
        
        // Use GIANT to solve ridge regression
        var giant: GiantExact.Driver = new GiantExact.Driver(sc, data, true)
        
        
        var error: Double = giant.predict(dataTest)
        println("Test error is " + error.toString)
        
        val results = giant.train(gamma, maxiter)
        
        println("\n ")
        println("Objective values are ")
        results._2.foreach(println)
        println("\n ")
        println("Training errors are ")
        results._1.foreach(println)
        println("\n ")
        println("Elapsed times are ")
        results._3.foreach(println)
        
        
        error = giant.predict(dataTest)
        println("Test error is " + error.toString)
        
        
        spark.stop()
    }
    
    
}