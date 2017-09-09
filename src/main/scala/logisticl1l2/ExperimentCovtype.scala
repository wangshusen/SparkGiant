package distopt.logisticl1l2

// spark-core
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd._
// spark-sql
import org.apache.spark.sql.SparkSession
// others
import scala.math

import distopt.utils._

object ExperimentCovtype {
    def main(args: Array[String]) {
        // parse parameters from command line arguments
        val filename1: String = args(0).toString
        val filename2: String = args(1).toString
        val numFeatures: Int = args(2).toInt
        val numSplits: Int = args(3).toInt
        
        println("Training name: " + filename1)
        println("Testing name: " + filename2)
        println("Number of random features: " + numFeatures.toString)
        println("Number of splits: " + numSplits.toString)
        
        // launch Spark
        var t0 = System.nanoTime()
        val spark = (SparkSession
                      .builder()
                      .appName("Distributed Algorithms for Logistic Regression")
                      .config("spark.some.config.option", "some-value")
                      .getOrCreate())
        val sc = spark.sparkContext
        sc.setLogLevel("ERROR")
        var t1 = System.nanoTime()
        println("Time cost of starting Spark:  " + ((t1-t0)*1e-9).toString + "  seconds.")
        
        // load data
        var (dataTrain, dataTest) = this.loaddata(spark, filename1, filename2, numSplits, numFeatures)
        dataTrain = dataTrain.persist()
        dataTrain.count
        dataTest = dataTest.persist()
        dataTest.count
        
        
        
        var gamma: Double = 1E-4
        //this.trainTestFista(gamma, sc, dataTrain, dataTest)
        this.trainTestGiant(gamma, sc, dataTrain, dataTest)
        
        
        spark.stop()
    }
    
    
    
    def trainTestFista(gamma: Double, sc: SparkContext, dataTrain: RDD[(Double, Array[Double])], dataTest: RDD[(Double, Array[Double])]): Unit = {
        val fista: Fista.Driver = new Fista.Driver(sc, dataTrain)
        val gamma2: Double = 1E-6
        
        var maxIterOuter: Int = 3000
        
        var lipchitz: Double = 100
        
        var results: (Array[Double], Array[Double], Array[Double]) = fista.train(gamma, gamma2, maxIterOuter, lipchitz)
        println("\n ")
        println("====================================================================")
        println("Fista (gamma1=" + gamma.toString + ", gamma2=" + gamma2.toString + ", MaxIterOuter=" + maxIterOuter.toString + ", lipchitz=" + lipchitz.toString + ")")
        println("\n ")
        println("Objective Value\t Training Error\t Elapsed Time")
        results.zipped.foreach(this.printAsTable)
        var testError: Double = fista.predict(dataTest)
        println("\n ")
        println("Test error is " + testError.toString)
        println("\n ")
        
        /*
        lipchitz = 1.0
        
        results = fista.train(gamma, gamma2, maxIterOuter, lipchitz)
        println("\n ")
        println("====================================================================")
        println("Fista (gamma1=" + gamma.toString + ", gamma2=" + gamma2.toString + ", MaxIterOuter=" + maxIterOuter.toString + ", lipchitz=" + lipchitz.toString + ")")
        println("\n ")
        println("Objective Value\t Training Error\t Elapsed Time")
        results.zipped.foreach(this.printAsTable)
        testError = fista.predict(dataTest)
        println("\n ")
        println("Test error is " + testError.toString)
        println("\n ")
        */
    }
    
    
    def trainTestGiant(gamma: Double, sc: SparkContext, dataTrain: RDD[(Double, Array[Double])], dataTest: RDD[(Double, Array[Double])]): Unit = {
        val giant: Giant.Driver = new Giant.Driver(sc, dataTrain)
        val gamma2: Double = 1E-6
        
        var maxIterOuter: Int = 30
        var maxIterInner: Int = 10
        
        var lipchitz: Double = 0.1
        
        var results: (Array[Double], Array[Double], Array[Double]) = giant.train(gamma, gamma2, maxIterOuter, maxIterInner, lipchitz)
        println("\n ")
        println("====================================================================")
        println("Giant (gamma1=" + gamma.toString + ", gamma2=" + gamma2.toString + ", MaxIterOuter=" + maxIterOuter.toString + ", MaxIterInner=" + maxIterInner.toString + ", lipchitz=" + lipchitz.toString + ")")
        println("\n ")
        println("Objective Value\t Training Error\t Elapsed Time")
        results.zipped.foreach(this.printAsTable)
        var testError: Double = giant.predict(dataTest)
        println("\n ")
        println("Test error is " + testError.toString)
        println("\n ")
        
        /*
        lipchitz = 1.0
        
        results = giant.train(gamma, gamma2, maxIterOuter, maxIterInner, lipchitz)
        println("\n ")
        println("====================================================================")
        println("Giant (gamma1=" + gamma.toString + ", gamma2=" + gamma2.toString + ", MaxIterOuter=" + maxIterOuter.toString + ", MaxIterInner=" + maxIterInner.toString + ", lipchitz=" + lipchitz.toString + ")")
        println("\n ")
        println("Objective Value\t Training Error\t Elapsed Time")
        results.zipped.foreach(this.printAsTable)
        testError = giant.predict(dataTest)
        println("\n ")
        println("Test error is " + testError.toString)
        println("\n ")
        */
    }
    
    def printAsTable(element1: Double, element2: Double, element3: Double): Unit = {
        println(element2.toString + "\t" + element1.toString + "\t" + element3.toString)
    }
    
    
    /**
     * Load training and testing data from lib-svm files.
     * 
     * @param spark Spark session
     * @param filename1 path of training data file
     * @param filename2 path of testing data file
     * @param numSplits number of splits
     * @return rdds of training and testing data
    */
    def loaddata(spark: SparkSession, filename1: String, filename2: String, numSplits: Int, numFeatures: Int): (RDD[(Double, Array[Double])], RDD[(Double, Array[Double])]) = {
        val t1 = System.nanoTime()
        
        // load training and test data
        val isCoalesce: Boolean = false
        var dataTrain: RDD[(Double, Array[Double])] = Utils.loadLibsvmData(spark, filename1, numSplits, isCoalesce)
                                                        .map(pair => (pair._1.toDouble * 2 - 3, pair._2))
                                                        .persist()
        var dataTest: RDD[(Double, Array[Double])] = Utils.loadLibsvmData(spark, filename2)
                                                        .map(pair => (pair._1.toDouble * 2 - 3, pair._2))
                                                        .persist()
        println("There are " + dataTrain.count.toString + " training samples.")
        println("There are " + dataTest.count.toString + " test samples.")
        val t2 = System.nanoTime()
        println("Time cost of loading data:  " + ((t2-t1)*1e-9).toString + "  seconds.")
        
        // estimate the kernel parameter (if it is unknown)
        //var sigma: Double = dataTrain.glom.map(Kernel.estimateSigma).mean
        //sigma = math.sqrt(sigma)
        //println("Estimated sigma is " + sigma.toString)
        
        /*
        // map input data to random Fourier features
        val sigmaCovtype: Double = 1.9
        dataTrain = dataTrain.mapPartitions(Kernel.rbfRfm(_, numFeatures, sigmaCovtype)).persist
        dataTest = dataTest.mapPartitions(Kernel.rbfRfm(_, numFeatures, sigmaCovtype)).persist
        println("There are " + dataTrain.count.toString + " training samples.")
        println("There are " + dataTest.count.toString + " test samples.")
        var t3 = System.nanoTime()
        println("Time cost of random feature mapping:  " + ((t3-t2)*1e-9).toString + "  seconds.")
        
        println("####################################")
        println("spark.conf.getAll:")
        spark.conf.getAll.foreach(println)
        println(" ")
        println("getExecutorMemoryStatus:")
        val sc: SparkContext = spark.sparkContext
        println(sc.getExecutorMemoryStatus.toString())
        println("####################################")
        println(" ")
        */
        (dataTrain, dataTest)
    }
}
