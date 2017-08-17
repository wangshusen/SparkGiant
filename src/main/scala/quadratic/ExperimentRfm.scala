package distopt.quadratic

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
import distopt.quadratic._

object ExperimentRfm {
    def main(args: Array[String]) {
        // parse parameters from command line arguments
        val filename1: String = args(0).toString
        val filename2: String = args(1).toString
        val numSplits: Int = args(2).toInt
        val numFeatures: Int = args(3).toInt
        
        println("Training file name: " + filename1)
        println("Test file name: " + filename2)
        println("Number of splits: " + numSplits.toString)
        
        // launch Spark
        var t0 = System.nanoTime()
        val spark = (SparkSession
                      .builder()
                      .appName("Distributed Algorithms for Logistic Regression")
                      .config("spark.some.config.option", "some-value")
                      .getOrCreate())
        val sc: SparkContext = spark.sparkContext
        sc.setLogLevel("ERROR")
        var t1 = System.nanoTime()
        println("Time cost of starting Spark:  " + ((t1-t0)*1e-9).toString + "  seconds.")
        
        // load data
        var (dataTrain, dataTest) = this.loaddata(spark, filename1, filename2, numSplits, numFeatures)
        
        
        var gamma: Double = 1E-4
        //this.trainTestGiant(gamma, sc, dataTrain, dataTest)
        this.trainTestAdmm(gamma, sc, dataTrain, dataTest)
        //this.trainTestCg(gamma, sc, dataTrain, dataTest)
        //this.trainTestLbfgs(gamma, sc, dataTrain, dataTest)
        
        spark.stop()
    }
    
    
    
    def trainTestGiant(gamma: Double, sc: SparkContext, dataTrain: RDD[(Double, Array[Double])], dataTest: RDD[(Double, Array[Double])]): Unit = {
        val isSearch: Boolean = true
        val giant: Giant.Driver = new Giant.Driver(sc, dataTrain, isSearch)
        
        var maxIterOuter: Int = 60
        var maxIterInner: Int = 100
        
        var results: (Array[Double], Array[Double], Array[Double]) = giant.train(gamma, maxIterOuter, maxIterInner)
        println("\n ")
        println("====================================================================")
        println("GIANT (gamma=" + gamma.toString + ", MaxIterOuter=" + maxIterOuter.toString + ", MaxIterInner=" + maxIterInner.toString + ")")
        println("\n ")
        println("Objective Value\t Training Error\t Elapsed Time")
        results.zipped.foreach(this.printAsTable)
        var testError: Double = giant.predict(dataTest)
        println("\n ")
        println("Test error is " + testError.toString)
        println("\n ")
        
        /*
        maxIterOuter = 30
        maxIterInner = 300
        
        results = giant.train(gamma, maxIterOuter, maxIterInner)
        println("\n ")
        println("====================================================================")
        println("GIANT (gamma=" + gamma.toString + ", MaxIterOuter=" + maxIterOuter.toString + ", MaxIterInner=" + maxIterInner.toString + ")")
        println("\n ")
        println("Objective Value\t Training Error\t Elapsed Time")
        results.zipped.foreach(this.printAsTable)
        testError = giant.predict(dataTest)
        println("\n ")
        println("Test error is " + testError.toString)
        println("\n ")
        */
    }
    
    def trainTestAdmm(gamma: Double, sc: SparkContext, dataTrain: RDD[(Double, Array[Double])], dataTest: RDD[(Double, Array[Double])]): Unit = {
        val admm: Admm.Driver = new Admm.Driver(sc, dataTrain)
        
        var maxIterOuter: Int = 100
        var maxIterInner: Int = 500
        
        var results: (Array[Double], Array[Double], Array[Double]) = admm.train(gamma, maxIterOuter, maxIterInner)
        println("\n ")
        println("====================================================================")
        println("ADMM (gamma=" + gamma.toString + ", MaxIterOuter=" + maxIterOuter.toString + ", MaxIterInner=" + maxIterInner.toString + ")")
        println("\n ")
        println("Objective Value\t Training Error\t Elapsed Time")
        results.zipped.foreach(this.printAsTable)
        var testError: Double = admm.predict(dataTest)
        println("\n ")
        println("Test error is " + testError.toString)
        println("\n ")
    }
    
    
    def trainTestCg(gamma: Double, sc: SparkContext, dataTrain: RDD[(Double, Array[Double])], dataTest: RDD[(Double, Array[Double])]): Unit = {
        val cg: Cg.Driver = new Cg.Driver(sc, dataTrain)
        
        var maxIterOuter: Int = 500
        
        var results: (Array[Double], Array[Double], Array[Double]) = cg.train(gamma, maxIterOuter)
        println("\n ")
        println("====================================================================")
        println("CG (gamma=" + gamma.toString + ", MaxIterOuter=" + maxIterOuter.toString + ")")
        println("\n ")
        println("Objective Value\t Training Error\t Elapsed Time")
        results.zipped.foreach(this.printAsTable)
        var testError: Double = cg.predict(dataTest)
        println("\n ")
        println("Test error is " + testError.toString)
        println("\n ")
    }
    
    
    def trainTestLbfgs(gamma: Double, sc: SparkContext, dataTrain: RDD[(Double, Array[Double])], dataTest: RDD[(Double, Array[Double])]): Unit = {
        val lbfgs: Lbfgs.Driver = new Lbfgs.Driver(sc, dataTrain)
        
        var maxIterOuter: Int = 200
        var numHistory: Int = 100
        
        var results: (Array[Double], Array[Double], Array[Double]) = lbfgs.train(gamma, maxIterOuter, numHistory)
        println("\n ")
        println("====================================================================")
        println("L-BFGS (gamma=" + gamma.toString + ", MaxIterOuter=" + maxIterOuter.toString + ", NumHistory=" + numHistory.toString + ")")
        println("\n ")
        println("Objective Value\t Training Error\t Elapsed Time")
        results.zipped.foreach(this.printAsTable)
        var testError: Double = lbfgs.predict(dataTest)
        println("\n ")
        println("Test error is " + testError.toString)
        println("\n ")
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
                                                        .map(pair => (pair._1.toDouble, pair._2))
                                                        .persist()
        var dataTest: RDD[(Double, Array[Double])] = Utils.loadLibsvmData(spark, filename2)
                                                        .map(pair => (pair._1.toDouble, pair._2))
                                                        .persist()
        println("There are " + dataTrain.count.toString + " training samples.")
        println("There are " + dataTest.count.toString + " test samples.")
        
        // normlaize the data
        val (meanLabel, maxFeatures): (Double, Array[Double]) = Utils.meanAndMax(dataTrain)
        val sc: SparkContext = spark.sparkContext
        dataTrain = Utils.normalize(sc, dataTrain, meanLabel, maxFeatures)
        dataTest = Utils.normalize(sc, dataTest, meanLabel, maxFeatures)
        val t2 = System.nanoTime()
        println("Time cost of loading data:  " + ((t2-t1)*1e-9).toString + "  seconds.")
        
        
        // estimate the kernel parameter (if it is unknown)
        //val sigma: Double = dataTrain.glom.map(Kernel.estimateSigma).mean
        //println("Estimated sigma is " + sigma.toString)
        
        // random feature mapping
        val sigmaYearPred: Double = 0.5510948938723474 // YearPrediction Dataset
        dataTrain = dataTrain.mapPartitions(Kernel.rbfRfm(_, numFeatures, sigmaYearPred)).persist()
        dataTest = dataTest.mapPartitions(Kernel.rbfRfm(_, numFeatures, sigmaYearPred)).persist()
        println("There are " + dataTrain.count.toString + " training samples.")
        println("There are " + dataTest.count.toString + " test samples.")
        var t3 = System.nanoTime()
        println("Time cost of random feature mapping:  " + ((t3-t2)*1e-9).toString + "  seconds.")
        
        
        println("####################################")
        println("spark.conf.getAll:")
        spark.conf.getAll.foreach(println)
        println(" ")
        println("getExecutorMemoryStatus:")
        println(sc.getExecutorMemoryStatus.toString())
        println("####################################")
        println(" ")
        
        (dataTrain, dataTest)
    }
}