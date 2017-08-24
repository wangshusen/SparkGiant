package distopt.logistic

// spark-core
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd._
// spark-sql
import org.apache.spark.sql.SparkSession
// spark-mllib
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.linalg.{Vector, Vectors}
// others
import scala.math

import distopt.utils._
import distopt.logistic._

object ExperimentMnist {
    def main(args: Array[String]) {
        // parse parameters from command line arguments
        val filename1: String = args(0).toString
        val filename2: String = args(1).toString
        val numFeatures: Int = args(2).toInt
        val numSplits: Int = args(3).toInt
        val d: Int = 784
        
        println("Training name: " + filename1)
        println("Testing name: " + filename2)
        println("Number of random features: " + numFeatures.toString)
        println("Number of splits: " + numSplits.toString)
        
        // launch Spark
        var t0 = System.nanoTime()
        val spark = (SparkSession
                      .builder()
                      .appName("Distributed Algorithms for Logistic Regression -- MNIST8K")
                      .config("spark.some.config.option", "some-value")
                      .getOrCreate())
        val sc = spark.sparkContext
        sc.setLogLevel("ERROR")
        var t1 = System.nanoTime()
        println("Time cost of starting Spark:  " + ((t1-t0)*1e-9).toString + "  seconds.")
        
        
        // partition data to train and test
        //val inputFileName: String = "./data/mnist"
        //partitionTrainTest(sc, inputFileName, numSplits)
        
        
        // load data
        var (dataTrain, dataTest) = this.loaddata(spark, filename1, filename2, d, numSplits, numFeatures)
        dataTrain = dataTrain.persist()
        dataTrain.count
        dataTest = dataTest.persist()
        dataTest.count
        
        
        // test logistic regression solvers
        var gamma: Double = 1E-6
        //this.trainTestGiant(gamma, sc, dataTrain, dataTest)
        this.trainTestDane(gamma, sc, dataTrain, dataTest)
        this.trainTestAdmm(gamma, sc, dataTrain, dataTest)
        this.trainTestAgd(gamma, sc, dataTrain, dataTest)
        this.trainTestLbfgs(gamma, sc, dataTrain, dataTest)
        
        /**/
        
        spark.stop()
    }
    
    
    
    def trainTestGiant(gamma: Double, sc: SparkContext, dataTrain: RDD[(Double, Array[Double])], dataTest: RDD[(Double, Array[Double])]): Unit = {
        val isSearch: Boolean = true
        val giant: Giant.Driver = new Giant.Driver(sc, dataTrain, isSearch)
        
        var maxIterOuter: Int = 100
        var maxIterInner: Int = 30
        
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
        
        maxIterOuter = 60
        maxIterInner = 100
        
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
        
        
        
        maxIterOuter = 20
        maxIterInner = 100
        
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
        
        
        maxIterOuter = 10
        maxIterInner = 300
        
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
        /**/
    }
    

    
    def trainTestAdmm(gamma: Double, sc: SparkContext, dataTrain: RDD[(Double, Array[Double])], dataTest: RDD[(Double, Array[Double])]): Unit = {
        val admm: Admm.Driver = new Admm.Driver(sc, dataTrain)
        
        
        var learningrate = 1.0
        
        var rho = 0.1
        
        var maxIterOuter = 40
        var maxIterInner = 30                                                                                                                  
        var results: (Array[Double], Array[Double], Array[Double]) = admm.train(gamma, maxIterOuter, maxIterInner, learningrate, rho)
        println("\n ")
        println("====================================================================")
        println("ADMM (gamma=" + gamma.toString + ", MaxIterOuter=" + maxIterOuter.toString + ", MaxIterInner=" + maxIterInner.toString + ", LearningRate=" + learningrate.toString + ", rho=" + rho.toString + ")")
        println("\n ")
        println("Objective Value\t Training Error\t Elapsed Time")
        results.zipped.foreach(this.printAsTable)
        var testError: Double = admm.predict(dataTest)
        println("\n ")
        println("Test error is " + testError.toString)
        println("\n ")
        
        maxIterOuter = 20
        maxIterInner = 100                                                                                                                  
        results = admm.train(gamma, maxIterOuter, maxIterInner, learningrate, rho)
        println("\n ")
        println("====================================================================")
        println("ADMM (gamma=" + gamma.toString + ", MaxIterOuter=" + maxIterOuter.toString + ", MaxIterInner=" + maxIterInner.toString + ", LearningRate=" + learningrate.toString + ", rho=" + rho.toString + ")")
        println("\n ")
        println("Objective Value\t Training Error\t Elapsed Time")
        results.zipped.foreach(this.printAsTable)
        testError = admm.predict(dataTest)
        println("\n ")
        println("Test error is " + testError.toString)
        println("\n ")
        
        
        maxIterOuter = 10
        maxIterInner = 300                                                                                                                  
        results = admm.train(gamma, maxIterOuter, maxIterInner, learningrate, rho)
        println("\n ")
        println("====================================================================")
        println("ADMM (gamma=" + gamma.toString + ", MaxIterOuter=" + maxIterOuter.toString + ", MaxIterInner=" + maxIterInner.toString + ", LearningRate=" + learningrate.toString + ", rho=" + rho.toString + ")")
        println("\n ")
        println("Objective Value\t Training Error\t Elapsed Time")
        results.zipped.foreach(this.printAsTable)
        testError = admm.predict(dataTest)
        println("\n ")
        println("Test error is " + testError.toString)
        println("\n ")
        
    }
    
    
    
    def trainTestAgd(gamma: Double, sc: SparkContext, dataTrain: RDD[(Double, Array[Double])], dataTest: RDD[(Double, Array[Double])]): Unit = {
        val agd: Agd.Driver = new Agd.Driver(sc, dataTrain)
        
        var maxIterOuter = 500
        
        var learningrate = 1.0
        var momentum = 0.95
        
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
        
        var numHistory: Int = 30
        
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
        
        numHistory = 100
        
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
    
    def partitionTrainTest(sc: SparkContext, filename: String, numSplits: Int): Unit = {
        // keep the samples with lable 4 or 9
        def filterFunction(a: Int): Boolean = { a == 4 || a == 9}
        var rawdata: RDD[String] = sc.textFile(filename)
                                .map(str => (str.slice(0, 2).toDouble.toInt, str.drop(2)))
                                .filter(pair => filterFunction(pair._1))
                                .map(pair => pair._1.toString + " " + pair._2)
                                .repartition(numSplits)
        // partition data to training and testing sets
        var indexedData: RDD[(String, Long)] = rawdata.zipWithIndex.persist()
        var trainData: RDD[String] = indexedData.filter(pair => (pair._2 % 5 > 0.1)).map(pair => pair._1)
        var testData: RDD[String] = indexedData.filter(pair => (pair._2 % 5 < 0.1)).map(pair => pair._1)
        trainData.saveAsTextFile(filename + "_train")
        testData.saveAsTextFile(filename + "_test")
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
    def loaddata(spark: SparkSession, filename1: String, filename2: String, d: Int, numSplits: Int, numFeatures: Int): (RDD[(Double, Array[Double])], RDD[(Double, Array[Double])]) = {
        val t1 = System.nanoTime()
        
        // load training and testing data
        val sc: SparkContext = spark.sparkContext
        var dataTrain: RDD[(Double, Array[Double])] = sc.textFile(filename1)
                                                        .repartition(numSplits)
                                                        .map(Utils.parseLibsvm(_, d))
                                                        .map(pair => ((pair._1.toDouble - 6.5) / 2.5, pair._2.toArray))
                                                        .persist()
        var dataTest: RDD[(Double, Array[Double])] = sc.textFile(filename2)
                                                        .map(Utils.parseLibsvm(_, d))
                                                        .map(pair => ((pair._1.toDouble - 6.5) / 2.5, pair._2.toArray))
                                                        .persist()
        println("There are " + dataTrain.count.toString + " training samples.")
        println("There are " + dataTest.count.toString + " test samples.")
        val t2 = System.nanoTime()
        println("Time cost of loading data:  " + ((t2-t1)*1e-9).toString + "  seconds.")
        //dataTrain.take(10).foreach(x => println(x._1.toString + " " + x._2.mkString(",")))
        
        
        // estimate the kernel parameter (if it is unknown)
        //var sigma: Double = dataTrain.glom.map(Kernel.estimateSigma).mean
        //sigma = math.sqrt(sigma)
        //println("Estimated sigma is " + sigma.toString)
        
        
        
        // map input data to random Fourier features
        val sigmaMnist: Double = 9.2
        dataTrain = dataTrain.mapPartitions(Kernel.rbfRfm(_, numFeatures, sigmaMnist)).persist
        dataTest = dataTest.mapPartitions(Kernel.rbfRfm(_, numFeatures, sigmaMnist)).persist
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
