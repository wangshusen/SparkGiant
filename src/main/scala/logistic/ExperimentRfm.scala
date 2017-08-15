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
                      .appName("Distributed Algorithms for Logistic Regression")
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
        
        
        
        var gamma: Double = 1E-4
        this.trainTestGiant(gamma, sc, dataTrain, dataTest)
        this.trainTestDane(gamma, sc, dataTrain, dataTest)
        this.trainTestAdmm(gamma, sc, dataTrain, dataTest)
        this.trainTestAgd(gamma, sc, dataTrain, dataTest)
        this.trainTestLbfgs(gamma, sc, dataTrain, dataTest)
        
        
        gamma = 1E-6
        this.trainTestGiant(gamma, sc, dataTrain, dataTest)
        this.trainTestDane(gamma, sc, dataTrain, dataTest)
        this.trainTestAdmm(gamma, sc, dataTrain, dataTest)
        this.trainTestAgd(gamma, sc, dataTrain, dataTest)
        this.trainTestLbfgs(gamma, sc, dataTrain, dataTest)
        
        
        gamma = 1E-8
        this.trainTestGiant(gamma, sc, dataTrain, dataTest)
        this.trainTestDane(gamma, sc, dataTrain, dataTest)
        this.trainTestAdmm(gamma, sc, dataTrain, dataTest)
        this.trainTestAgd(gamma, sc, dataTrain, dataTest)
        this.trainTestLbfgs(gamma, sc, dataTrain, dataTest)

        
        gamma = 1E-10
        this.trainTestGiant(gamma, sc, dataTrain, dataTest)
        this.trainTestDane(gamma, sc, dataTrain, dataTest)
        this.trainTestAdmm(gamma, sc, dataTrain, dataTest)
        this.trainTestAgd(gamma, sc, dataTrain, dataTest)
        this.trainTestLbfgs(gamma, sc, dataTrain, dataTest)
        
        spark.stop()
    }
    
    
    
    def trainTestGiant(gamma: Double, sc: SparkContext, dataTrain: RDD[(Double, Array[Double])], dataTest: RDD[(Double, Array[Double])]): Unit = {
        val isSearch: Boolean = true
        val giant: Giant.Driver = new Giant.Driver(sc, dataTrain, isSearch)
        
        
        var maxIterOuter: Int = 100
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
        
        
        maxIterOuter = 50
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
        
        
        var learningrate = 10.0
        
        var maxIterOuter = 30
        var maxIterInner = 100
        
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
        
        
        maxIterOuter = 15
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
    }
    

    
    def trainTestAdmm(gamma: Double, sc: SparkContext, dataTrain: RDD[(Double, Array[Double])], dataTest: RDD[(Double, Array[Double])]): Unit = {
        val admm: Admm.Driver = new Admm.Driver(sc, dataTrain)
        
        
        var learningrate = 10.0
        
        var maxIterOuter = 30
        var maxIterInner = 100                                                                                                                  
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
        
        
        maxIterOuter = 15
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
    }
    
    
    
    def trainTestAgd(gamma: Double, sc: SparkContext, dataTrain: RDD[(Double, Array[Double])], dataTest: RDD[(Double, Array[Double])]): Unit = {
        val agd: Agd.Driver = new Agd.Driver(sc, dataTrain)
        
        var maxIterOuter = 1000
        
        var learningrate = 10.0
        var momentum = 0.9
        
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
        
        learningrate = 10.0
        momentum = 0.95
        
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
        
        var maxIterOuter: Int = 200
        
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
    }
    
    
    def printAsTable(element1: Double, element2: Double, element3: Double): Unit = {
        println(element2.toString + "\t" + element1.toString + "\t" + element3.toString)
    }
}