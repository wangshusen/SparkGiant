package distopt.logistic

// spark-core
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd._
// spark-sql
import org.apache.spark.sql.SparkSession
// breeze
import breeze.linalg._
import breeze.numerics._
// others
import scala.math
import java.io._


import distopt.utils._
import distopt.logistic._

object Experiment {
    def main(args: Array[String]) {
        // parse parameters from command line arguments
        val filename: String = args(0).toString
        val numSplits: Int = args(1).toInt
        var gamma: Double = args(2).toDouble
        val maxiter: Int = args(3).toInt
        val q: Int = args(4).toInt

        
        // launch Spark
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
        
        // load training data
        var dataRaw: RDD[(Double, Array[Double])] = Utils.loadLibsvmData(spark, filename, numSplits, false)
                                                        .map(pair => (pair._1.toDouble, pair._2))
        
        var dataIdx: RDD[((Double, Array[Double]), Long)] = dataRaw.map(pair => (pair._1 * 2 - 3, pair._2))
                                                                .zipWithIndex
                                                                .persist()
        
        var data: RDD[(Double, Array[Double])] = dataIdx.filter(pair => (pair._2 % 5 > 0.1))
                                                        .map(pair => pair._1)
        var dataTest: RDD[(Double, Array[Double])] = dataIdx.filter(pair => (pair._2 % 5 == 0))
                                                        .map(pair => pair._1)
        
        
        println("####################################")
        println("spark.conf.getAll:")
        spark.conf.getAll.foreach(println)
        println(" ")
        println("getExecutorMemoryStatus:")
        println(sc.getExecutorMemoryStatus.toString())
        println("####################################")
        println(" ")
        
        val samples = data.take(10).map(pair => pair._1.toString + ",  " + pair._2.mkString(" ")).foreach(println)
        
        val isSearch: Boolean = true
        var giant: GiantCg.Driver = new GiantCg.Driver(sc, data, isSearch)
        
        trainAndTest(gamma, maxiter, q, giant, dataTest)
        
        spark.stop()
    }
    
    def trainAndTest(gamma: Double, maxiter: Int, q: Int, giant: GiantCg.Driver, dataTest: RDD[(Double, Array[Double])]): Unit = {
        val results = giant.train(gamma, maxiter, q)
        println("\n ")
        println("Objective values are ")
        results._2.foreach(println)
        println("\n ")
        println("Training errors are ")
        results._1.foreach(println)
        println("\n ")
        println("Elapsed times are ")
        results._3.foreach(println)
        
        val testError: Double = giant.predict(dataTest)
        println("\n ")
        println("Test error is " + testError.toString)
        println("\n ")
    }
    
    def loadData(spark: SparkSession, filename: String, numSplits: Int): (RDD[(Double, Array[Double])], RDD[(Double, Array[Double])]) = {
        val sc: SparkContext = spark.sparkContext
        
        // load training data
        var t0 = System.nanoTime()
        val dataRaw: RDD[(Double, Array[Double])] = Utils.loadLibsvmData(spark, filename, numSplits)
                                                        .map(pair => (pair._1.toDouble, pair._2))
                                                        .persist()
        println("Number of samples: " + dataRaw.count.toString)
        println("Number of partitions: " + dataRaw.getNumPartitions.toString)
        var t1 = System.nanoTime()
        println("Time cost of loading data:  " + ((t1-t0)*1e-9).toString + "  seconds.")
        /*
        // normalize training data
        val sc: SparkContext = spark.sparkContext
        t0 = System.nanoTime()
        val (meanLabel, maxFeatures) = Utils.meanAndMax(dataRaw)
        val data: RDD[(Double, Array[Double])] = Utils.normalize(sc, dataRaw, meanLabel, maxFeatures).persist()
        val n: Long = data.count
        t1 = System.nanoTime()
        println("n = " + n.toString)
        println("Time cost of data normalization:  " + ((t1-t0)*1e-9).toString + "  seconds.")
        println(" ")
        */
        
        // load and normalize test data
        val dataTestRaw: RDD[(Double, Array[Double])] = Utils.loadLibsvmData(spark, filename+".t")
                                                        .map(pair => (pair._1.toDouble, pair._2))
                                                        .persist()
        /*
        val dataTest: RDD[(Double, Array[Double])] = Utils.normalize(sc, dataTestRaw, meanLabel, maxFeatures).persist()
        */
        
        println("####################################")
        println("spark.conf.getAll:")
        spark.conf.getAll.foreach(println)
        println(" ")
        println("getExecutorMemoryStatus:")
        println(sc.getExecutorMemoryStatus.toString())
        println("####################################")
        println(" ")
        
        (dataRaw, dataTestRaw)
    }
    
}