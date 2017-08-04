package distopt.logistic

// spark-core
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd._
// spark-sql
import org.apache.spark.sql.SparkSession

import distopt.utils._
import distopt.logistic._

object ExperimentRfm {
    def main(args: Array[String]) {
        // parse parameters from command line arguments
        val filename: String = args(0).toString
        val numSplits: Int = args(1).toInt
        var gamma: Double = args(2).toDouble
        
        // launch Spark
        var t0 = System.nanoTime()
        val spark = (SparkSession
                      .builder()
                      .appName("Giant for Ridge Regression")
                      .config("spark.some.config.option", "some-value")
                      .getOrCreate())
        val sc = spark.sparkContext
        sc.setLogLevel("ERROR")
        var t1 = System.nanoTime()
        println("Time cost of starting Spark:  " + ((t1-t0)*1e-9).toString + "  seconds.")
        
        // load training data
        val isCoalesce: Boolean = false
        var dataRaw: RDD[(Double, Array[Double])] = Utils.loadLibsvmData(spark, filename, numSplits, isCoalesce)
                                                        .map(pair => (pair._1.toDouble, pair._2))
        
        // split to train and test
        var dataIdx: RDD[((Double, Array[Double]), Long)] = dataRaw.map(pair => (pair._1 * 2 - 3, pair._2))
                                                                .zipWithIndex
                                                                .persist()
        var dataRawTrain: RDD[(Double, Array[Double])] = dataIdx.filter(pair => (pair._2 % 5 > 0.1))
                                                        .map(pair => pair._1).persist()
        var dataRawTest: RDD[(Double, Array[Double])] = dataIdx.filter(pair => (pair._2 % 5 == 0))
                                                        .map(pair => pair._1)
        
        // estimate the kernel parameter (if it is unknown)
        //val sigma: Double = dataRawTrain.glom.map(Kernel.estimateSigma).mean
        //println("Estimated sigma is " + sigma.toString)
        
        // map input data to random Fourier features
        val numFeature: Int = 100
        val sigmaCovtype: Double = 3.2
        var dataTrain: RDD[(Double, Array[Double])] = dataRawTrain.mapPartitions(Kernel.rbfRfm(_, numFeature, sigmaCovtype))
        var dataTest: RDD[(Double, Array[Double])] = dataRawTest.mapPartitions(Kernel.rbfRfm(_, numFeature, sigmaCovtype))
        
        
        println("####################################")
        println("spark.conf.getAll:")
        spark.conf.getAll.foreach(println)
        println(" ")
        println("getExecutorMemoryStatus:")
        println(sc.getExecutorMemoryStatus.toString())
        println("####################################")
        println(" ")
        
        
        // GIANT
        var maxIterOuter: Int = 10
        var maxIterInner: Int = 100
        var isSearch: Boolean = true
        var giant: GiantCg.Driver = new GiantCg.Driver(sc, dataTrain, isSearch)
        trainTestGiant(gamma, maxIterOuter, maxIterInner, giant, dataTest)
        
        // DANE
        isSearch = true
        var learningrate: Double = 10.0
        var dane: Dane.Driver = new Dane.Driver(sc, dataTrain, isSearch)
        trainTestDane(gamma, maxIterOuter, maxIterInner, learningrate, dane, dataTest)
        
        
        spark.stop()
    }
    
    /**
     * @param gamma regularization parameter
     * @param maxiter max number of iterations (outer loop)
     * @param q max number of iterations (inner loop)
     * @param giant Giant object
     * @param dataTest RDD of test label-vector pairs
     */
    def trainTestGiant(gamma: Double, maxiter: Int, q: Int, giant: GiantCg.Driver, dataTest: RDD[(Double, Array[Double])]): Unit = {
        val results = giant.train(gamma, maxiter, q)
        println("\n ")
        println("GIANT: ")
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
    
    /**
     * @param gamma regularization parameter
     * @param maxiter max number of iterations (outer loop)
     * @param q max number of iterations (inner loop)
     * @param learningrate learning rate
     * @param dane Dane object
     * @param dataTest RDD of test label-vector pairs
     */
    def trainTestDane(gamma: Double, maxiter: Int, q: Int, learningrate: Double, dane: Dane.Driver, dataTest: RDD[(Double, Array[Double])]): Unit = {
        val results = dane.train(gamma, maxiter, q, learningrate)
        println("\n ")
        println("DANE: ")
        println("\n ")
        println("Objective values are ")
        results._2.foreach(println)
        println("\n ")
        println("Training errors are ")
        results._1.foreach(println)
        println("\n ")
        println("Elapsed times are ")
        results._3.foreach(println)
        
        val testError: Double = dane.predict(dataTest)
        println("\n ")
        println("Test error is " + testError.toString)
        println("\n ")
    }
    
}