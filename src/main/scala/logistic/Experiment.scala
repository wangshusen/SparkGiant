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

object Experiment {
    def main(args: Array[String]) {
        // parse parameters from command line arguments
        val filename1: String = args(0).toString
        val filename2: String = args(1).toString
        val numSplits: Int = args(2).toInt
        
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
        val sc = spark.sparkContext
        sc.setLogLevel("ERROR")
        var t1 = System.nanoTime()
        println("Time cost of starting Spark:  " + ((t1-t0)*1e-9).toString + "  seconds.")
        
        // load training and test data
        val isCoalesce: Boolean = false
        val dataTrain: RDD[(Double, Array[Double])] = Utils.loadLibsvmData(spark, filename1, numSplits, isCoalesce)
                                                        .map(pair => (pair._1.toDouble, pair._2))
                                                        .persist()
        val dataTest: RDD[(Double, Array[Double])] = Utils.loadLibsvmData(spark, filename2)
                                                        .map(pair => (pair._1.toDouble, pair._2))
                                                        .persist()
        
        println("There are " + dataTrain.count.toString + " training samples.")
        println("There are " + dataTest.count.toString + " test samples.")
        var t2 = System.nanoTime()
        println("Time cost of loading data:  " + ((t2-t1)*1e-9).toString + "  seconds.")
        
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
        //this.trainTestDane(gamma, sc, dataTrain, dataTest)
        //this.trainTestAdmm(gamma, sc, dataTrain, dataTest)
        //this.trainTestAgd(gamma, sc, dataTrain, dataTest)
        //this.trainTestLbfgs(gamma, sc, dataTrain, dataTest)

        /*
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
        
        */
        
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
    
    
    def trainTestDane(gamma: Double, sc: SparkContext, dataTrain: RDD[(Double, Array[Double])], dataTest: RDD[(Double, Array[Double])]): Unit = {
        val isSearch = true
        val dane: Dane.Driver = new Dane.Driver(sc, dataTrain, isSearch)
        
        
        var learningrate = 10.0
        
        var maxIterOuter = 20
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
        
        /*
        maxIterOuter = 10
        maxIterInner = 200
        
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
        */
    }
    

    
    def trainTestAdmm(gamma: Double, sc: SparkContext, dataTrain: RDD[(Double, Array[Double])], dataTest: RDD[(Double, Array[Double])]): Unit = {
        val admm: Admm.Driver = new Admm.Driver(sc, dataTrain)
        
        
        var learningrate = 10.0
        
        var maxIterOuter = 20
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
        
        /*
        maxIterOuter = 10
        maxIterInner = 200                                                                                                                  
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
        
        var maxIterOuter = 2000
        
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