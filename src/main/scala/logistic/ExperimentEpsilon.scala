package distopt.logistic

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
import distopt.logistic._

object ExperimentEpsilon {
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
                      .appName("Distributed Algorithms for Logistic Regression -- Epsilon")
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
        //this.trainTestGiant(gamma, sc, dataTrain, dataTest)
        //this.trainTestDane(gamma, sc, dataTrain, dataTest)
        //this.trainTestAdmm(gamma, sc, dataTrain, dataTest)
        this.trainTestAgd(gamma, sc, dataTrain, dataTest)
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
        
        
        maxIterOuter = 15
        maxIterInner = 900
        
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
        
    }
    
    
    def trainTestDane(gamma: Double, sc: SparkContext, dataTrain: RDD[(Double, Array[Double])], dataTest: RDD[(Double, Array[Double])]): Unit = {
        val isSearch = true
        val dane: Dane.Driver = new Dane.Driver(sc, dataTrain, isSearch)
        
        
        var learningrate = 1.0
        
        var maxIterOuter = 40
        var maxIterInner = 30
        
        var results: (Array[Double], Array[Double], Array[Double]) = dane.train(gamma, maxIterOuter, maxIterInner, learningrate)
        println("\n ")
        println("====================================================================")
        println("DANE (gamma=" + gamma.toString + ", MaxIterOuter=" + maxIterOuter.toString + ", MaxIterInner=" + maxIterInner.toString + ", LearningRate=" + learningrate.toString + ")")
        println("\n ")
        println("Objective Value\t Training Error\t Elapsed Time")
        results.zipped.foreach(this.printAsTable)
        var testError: Double = dane.predict(dataTest)
        println("\n ")
        println("Test error is " + testError.toString)
        println("\n ")
        
        learningrate = 10.0
        maxIterOuter = 20
        maxIterInner = 30
        
        results = dane.train(gamma, maxIterOuter, maxIterInner, learningrate)
        println("\n ")
        println("====================================================================")
        println("DANE (gamma=" + gamma.toString + ", MaxIterOuter=" + maxIterOuter.toString + ", MaxIterInner=" + maxIterInner.toString + ", LearningRate=" + learningrate.toString + ")")
        println("\n ")
        println("Objective Value\t Training Error\t Elapsed Time")
        results.zipped.foreach(this.printAsTable)
        testError = dane.predict(dataTest)
        println("\n ")
        println("Test error is " + testError.toString)
        println("\n ")
        
        learningrate = 100.0
        maxIterOuter = 20
        maxIterInner = 30
        
        results = dane.train(gamma, maxIterOuter, maxIterInner, learningrate)
        println("\n ")
        println("====================================================================")
        println("DANE (gamma=" + gamma.toString + ", MaxIterOuter=" + maxIterOuter.toString + ", MaxIterInner=" + maxIterInner.toString + ", LearningRate=" + learningrate.toString + ")")
        println("\n ")
        println("Objective Value\t Training Error\t Elapsed Time")
        results.zipped.foreach(this.printAsTable)
        testError = dane.predict(dataTest)
        println("\n ")
        println("Test error is " + testError.toString)
        println("\n ")
        
    }
    

    
    def trainTestAdmm(gamma: Double, sc: SparkContext, dataTrain: RDD[(Double, Array[Double])], dataTest: RDD[(Double, Array[Double])]): Unit = {
        val admm: Admm.Driver = new Admm.Driver(sc, dataTrain)
        
        
        var learningrate = 10.0
        
        var maxIterOuter = 40
        var maxIterInner = 30                                                                                                                  
        var results: (Array[Double], Array[Double], Array[Double]) = admm.train(gamma, maxIterOuter, maxIterInner, learningrate)
        println("\n ")
        println("====================================================================")
        println("ADMM (gamma=" + gamma.toString + ", MaxIterOuter=" + maxIterOuter.toString + ", MaxIterInner=" + maxIterInner.toString + ", LearningRate=" + learningrate.toString + ")")
        println("\n ")
        println("Objective Value\t Training Error\t Elapsed Time")
        results.zipped.foreach(this.printAsTable)
        var testError: Double = admm.predict(dataTest)
        println("\n ")
        println("Test error is " + testError.toString)
        println("\n ")
        
        maxIterOuter = 20
        maxIterInner = 100                                                                                                                  
        results = admm.train(gamma, maxIterOuter, maxIterInner, learningrate)
        println("\n ")
        println("====================================================================")
        println("ADMM (gamma=" + gamma.toString + ", MaxIterOuter=" + maxIterOuter.toString + ", MaxIterInner=" + maxIterInner.toString + ", LearningRate=" + learningrate.toString + ")")
        println("\n ")
        println("Objective Value\t Training Error\t Elapsed Time")
        results.zipped.foreach(this.printAsTable)
        testError = admm.predict(dataTest)
        println("\n ")
        println("Test error is " + testError.toString)
        println("\n ")
        
        
        maxIterOuter = 10
        maxIterInner = 300                                                                                                                  
        results = admm.train(gamma, maxIterOuter, maxIterInner, learningrate)
        println("\n ")
        println("====================================================================")
        println("ADMM (gamma=" + gamma.toString + ", MaxIterOuter=" + maxIterOuter.toString + ", MaxIterInner=" + maxIterInner.toString + ", LearningRate=" + learningrate.toString + ")")
        println("\n ")
        println("Objective Value\t Training Error\t Elapsed Time")
        results.zipped.foreach(this.printAsTable)
        testError = admm.predict(dataTest)
        println("\n ")
        println("Test error is " + testError.toString)
        println("\n ")
        
        /*
        maxIterOuter = 5
        maxIterInner = 900                                                                                                                  
        results = admm.train(gamma, maxIterOuter, maxIterInner, learningrate)
        println("\n ")
        println("====================================================================")
        println("ADMM (gamma=" + gamma.toString + ", MaxIterOuter=" + maxIterOuter.toString + ", MaxIterInner=" + maxIterInner.toString + ", LearningRate=" + learningrate.toString + ")")
        println("\n ")
        println("Objective Value\t Training Error\t Elapsed Time")
        results.zipped.foreach(this.printAsTable)
        testError = admm.predict(dataTest)
        println("\n ")
        println("Test error is " + testError.toString)
        println("\n ")
        */
    }
    
    
    
    def trainTestAgd(gamma: Double, sc: SparkContext, dataTrain: RDD[(Double, Array[Double])], dataTest: RDD[(Double, Array[Double])]): Unit = {
        val agd: Agd.Driver = new Agd.Driver(sc, dataTrain)
        
        var maxIterOuter = 500
        
        var learningrate = 0.1
        var momentum = 0.99
        
        var results: (Array[Double], Array[Double], Array[Double]) = agd.train(gamma, maxIterOuter, learningrate, momentum)
        println("\n ")
        println("====================================================================")
        println("Accelerated Gradient Descent (gamma=" + gamma.toString + ", MaxIterOuter=" + maxIterOuter.toString+ ", LearningRate=" + learningrate.toString + ", momentum=" + momentum.toString + ")")
        println("\n ")
        println("Objective Value\t Training Error\t Elapsed Time")
        results.zipped.foreach(this.printAsTable)
        var testError: Double = agd.predict(dataTest)
        println("\n ")
        println("Test error is " + testError.toString)
        println("\n ")
        
        
        learningrate = 1.0
        momentum = 0.99
        
        results = agd.train(gamma, maxIterOuter, learningrate, momentum)
        println("\n ")
        println("====================================================================")
        println("Accelerated Gradient Descent (gamma=" + gamma.toString + ", MaxIterOuter=" + maxIterOuter.toString+ ", LearningRate=" + learningrate.toString + ", momentum=" + momentum.toString + ")")
        println("\n ")
        println("Objective Value\t Training Error\t Elapsed Time")
        results.zipped.foreach(this.printAsTable)
        testError = agd.predict(dataTest)
        println("\n ")
        println("Test error is " + testError.toString)
        println("\n ")
        
        
        learningrate = 10.0
        momentum = 0.99
        
        results = agd.train(gamma, maxIterOuter, learningrate, momentum)
        println("\n ")
        println("====================================================================")
        println("Accelerated Gradient Descent (gamma=" + gamma.toString + ", MaxIterOuter=" + maxIterOuter.toString+ ", LearningRate=" + learningrate.toString + ", momentum=" + momentum.toString + ")")
        println("\n ")
        println("Objective Value\t Training Error\t Elapsed Time")
        results.zipped.foreach(this.printAsTable)
        testError = agd.predict(dataTest)
        println("\n ")
        println("Test error is " + testError.toString)
        println("\n ")
        
        learningrate = 100.0
        momentum = 0.99
        
        results = agd.train(gamma, maxIterOuter, learningrate, momentum)
        println("\n ")
        println("====================================================================")
        println("Accelerated Gradient Descent (gamma=" + gamma.toString + ", MaxIterOuter=" + maxIterOuter.toString+ ", LearningRate=" + learningrate.toString + ", momentum=" + momentum.toString + ")")
        println("\n ")
        println("Objective Value\t Training Error\t Elapsed Time")
        results.zipped.foreach(this.printAsTable)
        testError = agd.predict(dataTest)
        println("\n ")
        println("Test error is " + testError.toString)
        println("\n ")
        
    }
    
    
    
    def trainTestLbfgs(gamma: Double, sc: SparkContext, dataTrain: RDD[(Double, Array[Double])], dataTest: RDD[(Double, Array[Double])]): Unit = {
        val lbfgs: Lbfgs.Driver = new Lbfgs.Driver(sc, dataTrain)
        
        var maxIterOuter: Int = 500
        
        var numHistory: Int = 100
        
        var results: (Array[Double], Array[Double], Array[Double]) = lbfgs.train(gamma, maxIterOuter, numHistory)
        println("\n ")
        println("====================================================================")
        println("L-BFGS (gamma=" + gamma.toString + ", MaxIterOuter=" + maxIterOuter.toString + ", numHistory=" + numHistory.toString + ")")
        println("\n ")
        println("Objective Value\t Training Error\t Elapsed Time")
        results.zipped.foreach(this.printAsTable)
        var testError: Double = lbfgs.predict(dataTest)
        println("\n ")
        println("Test error is " + testError.toString)
        println("\n ")
        
        
        numHistory = 300
        
        results = lbfgs.train(gamma, maxIterOuter, numHistory)
        println("\n ")
        println("====================================================================")
        println("L-BFGS (gamma=" + gamma.toString + ", MaxIterOuter=" + maxIterOuter.toString + ", numHistory=" + numHistory.toString + ")")
        println("\n ")
        println("Objective Value\t Training Error\t Elapsed Time")
        results.zipped.foreach(this.printAsTable)
        testError = lbfgs.predict(dataTest)
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
        val t2 = System.nanoTime()
        println("Time cost of loading data:  " + ((t2-t1)*1e-9).toString + "  seconds.")
        
        
        // estimate the kernel parameter (if it is unknown)
        //var sigma: Double = dataTrain.glom.map(Kernel.estimateSigma).mean
        //sigma = math.sqrt(sigma)
        //println("Estimated sigma is " + sigma.toString)
        
        
        // map input data to random Fourier features
        val sigmaEpsilon: Double = 1.3
        dataTrain = dataTrain.mapPartitions(Kernel.rbfRfm(_, numFeatures, sigmaEpsilon)).persist
        dataTest = dataTest.mapPartitions(Kernel.rbfRfm(_, numFeatures, sigmaEpsilon)).persist
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
        
        (dataTrain, dataTest)
    }
}
