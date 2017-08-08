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
        val numFeatures: Int = args(1).toInt
        val numSplits: Int = args(2).toInt
        
        println("File name: " + filename)
        println("Number of random features: " + numFeatures.toString)
        println("Number of splits: " + numSplits.toString)
        
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
                                                        .map(pair => pair._1).persist()
        println("There are " + dataRawTrain.count.toString + " training samples.")
        println("There are " + dataRawTest.count.toString + " test samples.")
        var t2 = System.nanoTime()
        println("Time cost of loading data:  " + ((t2-t1)*1e-9).toString + "  seconds.")
        
        // estimate the kernel parameter (if it is unknown)
        //val sigma: Double = dataRawTrain.glom.map(Kernel.estimateSigma).mean
        //println("Estimated sigma is " + sigma.toString)
        
        // map input data to random Fourier features
        val sigmaCovtype: Double = 3.2
        var dataTrain: RDD[(Double, Array[Double])] = dataRawTrain.mapPartitions(Kernel.rbfRfm(_, numFeatures, sigmaCovtype)).persist
        var dataTest: RDD[(Double, Array[Double])] = dataRawTest.mapPartitions(Kernel.rbfRfm(_, numFeatures, sigmaCovtype)).persist
        println("There are " + dataTrain.count.toString + " training executors.")
        println("There are " + dataTest.count.toString + " test executors.")
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
        
        
        //this.determineGamma(dataTrain, dataTest, sc)
        
        var gamma: Double = 1E-8
        this.experiment(gamma, dataTrain, dataTest, sc)
        
        
        spark.stop()
    }
    
    def determineGamma(dataTrain: RDD[(Double, Array[Double])], dataTest: RDD[(Double, Array[Double])], sc: SparkContext): Unit = {
        var isSearch: Boolean = true
        val giant: Giant.Driver = new Giant.Driver(sc, dataTrain, isSearch)
        
        val maxIterOuter: Int = 30
        val maxIterInner: Int = 100
        
        var gamma: Double = 1E-8
        println("#######################################################################")
        println("Regularization parameter gamma = " + gamma.toString)
        println("#######################################################################")
        trainTestGiant(gamma, maxIterOuter, maxIterInner, giant, dataTest)
        
        gamma = 1E-9
        println("#######################################################################")
        println("Regularization parameter gamma = " + gamma.toString)
        println("#######################################################################")
        //trainTestGiant(gamma, maxIterOuter, maxIterInner, giant, dataTest)
        
        gamma = 1E-10
        println("#######################################################################")
        println("Regularization parameter gamma = " + gamma.toString)
        println("#######################################################################")
        //trainTestGiant(gamma, maxIterOuter, maxIterInner, giant, dataTest)
    }
    
    def compare(gamma: Double, dataTrain: RDD[(Double, Array[Double])], dataTest: RDD[(Double, Array[Double])], sc: SparkContext): Unit = {
        var isSearch: Boolean = true
        var maxIterOuter: Int = 10
        var maxIterInner: Int = 100
        var learningrate: Double = 1.0
        var momentum: Double = 0.5
        
        // --------- Accelerated gradient descent --------- //
        val agd: Agd.Driver = new Agd.Driver(sc, dataTrain)
        
        maxIterOuter = 1000
        learningrate = 10.0
        /*
        momentum = 0.5
        trainTestAgd(gamma, maxIterOuter, learningrate, momentum, agd, dataTest)
        
        momentum = 0.9
        trainTestAgd(gamma, maxIterOuter, learningrate, momentum, agd, dataTest)
        */
        momentum = 0.95
        trainTestAgd(gamma, maxIterOuter, learningrate, momentum, agd, dataTest)
        
        /*
        // --------------------- GIANT --------------------- //
        isSearch = true
        val giant: Giant.Driver = new Giant.Driver(sc, dataTrain, isSearch)
        
        maxIterOuter = 100
        maxIterInner = 30
        trainTestGiant(gamma, maxIterOuter, maxIterInner, giant, dataTest)
        
        maxIterOuter = 50
        maxIterInner = 100
        trainTestGiant(gamma, maxIterOuter, maxIterInner, giant, dataTest)
        
        maxIterOuter = 25
        maxIterInner = 300
        trainTestGiant(gamma, maxIterOuter, maxIterInner, giant, dataTest)
        */
        
        // --------------------- DANE --------------------- //
        isSearch = true
        val dane: Dane.Driver = new Dane.Driver(sc, dataTrain, isSearch)
        
        learningrate = 10.0
        
        //maxIterOuter = 100
        //maxIterInner = 30
        //trainTestDane(gamma, maxIterOuter, maxIterInner, learningrate, dane, dataTest)
        
        maxIterOuter = 50
        maxIterInner = 100
        trainTestDane(gamma, maxIterOuter, maxIterInner, learningrate, dane, dataTest)
        
        //maxIterOuter = 25
        //maxIterInner = 300
        //trainTestDane(gamma, maxIterOuter, maxIterInner, learningrate, dane, dataTest)
        
        
        // --------------------- ADMM --------------------- //
        val admm: Admm.Driver = new Admm.Driver(sc, dataTrain)
        
        learningrate = 10.0
        
        //maxIterOuter = 200
        //maxIterInner = 30
        //trainTestAdmm(gamma, maxIterOuter, maxIterInner, learningrate, admm, dataTest)
        
        maxIterOuter = 100
        maxIterInner = 100
        trainTestAdmm(gamma, maxIterOuter, maxIterInner, learningrate, admm, dataTest)
        
        //maxIterOuter = 50
        //maxIterInner = 300
        //trainTestAdmm(gamma, maxIterOuter, maxIterInner, learningrate, admm, dataTest)
        
    }
    
    /**
     * @param gamma regularization parameter
     * @param maxiter max number of iterations (outer loop)
     * @param q max number of iterations (inner loop)
     * @param giant Giant object
     * @param dataTest RDD of test label-vector pairs
     */
    def trainTestGiant(gamma: Double, maxiter: Int, q: Int, giant: Giant.Driver, dataTest: RDD[(Double, Array[Double])]): Unit = {
        val results: (Array[Double], Array[Double], Array[Double]) = giant.train(gamma, maxiter, q)
        println("\n ")
        println("====================================================================")
        println("GIANT (gamma=" + gamma.toString + ", MaxIterOuter=" + maxiter.toString + ", MaxIterInner=" + q.toString + ")")
        println("\n ")
        
        println("Objective Value\t Training Error\t Elapsed Time")
        results.zipped.foreach(this.printAsTable)
        
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
        val results: (Array[Double], Array[Double], Array[Double]) = dane.train(gamma, maxiter, q, learningrate)
        println("\n ")
        println("====================================================================")
        println("DANE (gamma=" + gamma.toString + ", MaxIterOuter=" + maxiter.toString + ", MaxIterInner=" + q.toString + ", LearningRate=" + learningrate.toString + ")")
        println("\n ")
        
        println("Objective Value\t Training Error\t Elapsed Time")
        results.zipped.foreach(this.printAsTable)
        
        val testError: Double = dane.predict(dataTest)
        println("\n ")
        println("Test error is " + testError.toString)
        println("\n ")
    }
    
    
    /**
     * @param gamma regularization parameter
     * @param maxiter max number of iterations (outer loop)
     * @param learningrate learning rate
     * @param momentum (between 0 and 1)
     * @param agd Agd object
     * @param dataTest RDD of test label-vector pairs
     */
    def trainTestAgd(gamma: Double, maxiter: Int, learningrate: Double, momentum: Double, agd: Agd.Driver, dataTest: RDD[(Double, Array[Double])]): Unit = {
        val results: (Array[Double], Array[Double], Array[Double]) = agd.train(gamma, maxiter, learningrate, momentum)
        println("\n ")
        println("====================================================================")
        println("Accelerated Gradient Descent (gamma=" + gamma.toString + ", MaxIterOuter=" + maxiter.toString+ ", LearningRate=" + learningrate.toString + ", momentum=" + momentum.toString + ")")
        println("\n ")
        
        println("Objective Value\t Training Error\t Elapsed Time")
        results.zipped.foreach(this.printAsTable)
        
        val testError: Double = agd.predict(dataTest)
        println("\n ")
        println("Test error is " + testError.toString)
        println("\n ")
    }
    
    
    /**
     * @param gamma regularization parameter
     * @param maxiter max number of iterations (outer loop)
     * @param q max number of iterations (inner loop)
     * @param learningrate learning rate
     * @param admm Admm object
     * @param dataTest RDD of test label-vector pairs
     */
    def trainTestAdmm(gamma: Double, maxiter: Int, q: Int, learningrate: Double, admm: Admm.Driver, dataTest: RDD[(Double, Array[Double])]): Unit = {
        val results: (Array[Double], Array[Double], Array[Double]) = admm.train(gamma, maxiter, q, learningrate)
        println("\n ")
        println("====================================================================")
        println("ADMM (gamma=" + gamma.toString + ", MaxIterOuter=" + maxiter.toString + ", MaxIterInner=" + q.toString + ", LearningRate=" + learningrate.toString + ")")
        println("\n ")
        
        println("Objective Value\t Training Error\t Elapsed Time")
        results.zipped.foreach(this.printAsTable)
        
        
        val testError: Double = admm.predict(dataTest)
        println("\n ")
        println("Test error is " + testError.toString)
        println("\n ")
    }
    
    def printAsTable(element1: Double, element2: Double, element3: Double): Unit = {
        println(element2.toString + "\t" + element1.toString + "\t" + element3.toString)
    }
    
    /*
    def printAsTable(tuple: (Double, Double, Double)): Unit = {
        println(tuple._2.toString + "\t" + tuple._1.toString + "\t" + tuple._3.toString)
    }*/
}