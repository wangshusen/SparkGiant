package distopt.utils

// breeze
import breeze.linalg._
import breeze.numerics._
// others
import scala.math


object SdcaL1L2 {
    /**
     * Minimize the objective w.r.t. w:
     *      (0.5/s)*||X*w - y||_2^2 + lambda*(0.5*||w||_2^2 + sigma*||w||_1 - w'*z).
     *
     * @param x d-by-s feature matrix
     * @param y s-by-1 label vector
     * @param z d-by-1 vector in the above objective function
     * @param lambda parameter in the above objective function
     * @param sigma parameter in the above objective function
     * @param normalizers0 (optional) s-by-1 vector whose j-th entry is 1 + ||x_j||_2^2 / 2/ lambda / s
     * @param numEpochs (optional) number of epoches
     * @param alpha0 (optional) initial guess of alpha
     * @param w0 (optional) initial guess of w
     * @return (wbar, abar) the average of the primal-dual variables pair (w, alpha)
     */
    def sdcaQuadraticL1L2(x: DenseMatrix[Double], y: DenseVector[Double], z: DenseVector[Double], lambda: Double, sigma: Double, normalizers0: DenseVector[Double] = null, numEpochs: Int = 20, alpha0: DenseVector[Double] = null, w0: DenseVector[Double] = null): (DenseVector[Double], DenseVector[Double]) = {
        // other parameters
        val sizeBatch: Int = 5
        val isMute = true
        
        // constants
        val s: Int = x.cols
        val d: Int = x.rows
        val invSizeBatch: Double = 1.0 / sizeBatch
        val numLoop: Int = math.floor(s * invSizeBatch).toInt
        val invNumLoop: Double = 1.0 / numLoop.toDouble
        assert(y.length == s)
        val invLambdaS: Double = 1.0 / lambda / s.toDouble
        
        var normalizers: DenseVector[Double] = normalizers0
        if (normalizers0 == null){
            val xColNormsSquare = new Array[Double](s)
            val slambda2: Double = 2.0 * lambda * s
            for (j <- 0 until s) xColNormsSquare(j) = x(::, j).toArray.map(b => b*b).sum
            normalizers = new DenseVector(xColNormsSquare.map(b => 1.0 / (1.0 + b / slambda2)))
        }
                
        // compute sign(b) * max{0, abs(b) - gamma}
        def shrinkage(b: Double, gamma: Double): Double = {
            math.signum(b) * math.max(0.0, math.abs(b) - gamma)
        }
        
        // initialization
        val alpha: DenseVector[Double] = if (alpha0 != null) alpha0 else {
            if (w0 != null) (y - x.t * w0)
            else DenseVector.zeros[Double](s)
        }
        val v: DenseVector[Double] = invLambdaS * (x * alpha)
        val vz: DenseVector[Double] = v + z
        val w: DenseVector[Double] = DenseVector.zeros[Double](d)
        for (j <- 0 until d) w(j) = shrinkage(vz(j), sigma)
        
        for (t <- 0 until numEpochs-1) {
            val randIndex: List[Int] = scala.util.Random.shuffle((0 until (s-sizeBatch+1)).toList)
            for (i <- 0 until numLoop) {
                // sampling
                val idx: Int = randIndex(i)
                val r = idx until (idx+sizeBatch)
                val xSample: DenseMatrix[Double] = x(::, r)
                val ySample: DenseVector[Double] = y(r)
                val aSample: DenseVector[Double] = alpha(r)
                val nSample: DenseVector[Double] = normalizers(r) * invSizeBatch
                
                // update
                val delta: DenseVector[Double] = ySample - xSample.t * w - aSample
                for (j <- 0 until sizeBatch) delta(j) *= nSample(j)
                for (j <- 0 until sizeBatch) alpha(idx+j) += delta(j)
                v += xSample * (invLambdaS * delta)
                vz := v + z
                for (j <- 0 until d) w(j) = shrinkage(vz(j), sigma)
            }
        }
        
        // average options
        var wbar: DenseVector[Double] = DenseVector.zeros[Double](d)
        var abar: DenseVector[Double] = alpha.copy
        val randIndex: List[Int] = scala.util.Random.shuffle((0 until (s-sizeBatch+1)).toList)
        for (i <- 0 until numLoop) {
            // sampling
            val idx: Int = randIndex(i)
            val r = idx until (idx+sizeBatch)
            val xSample: DenseMatrix[Double] = x(::, r)
            val ySample: DenseVector[Double] = y(r)
            val aSample: DenseVector[Double] = alpha(r)
            val nSample: DenseVector[Double] = normalizers(r) * invSizeBatch

            // update
            val delta: DenseVector[Double] = ySample - xSample.t * w - aSample
            for (j <- 0 until sizeBatch) delta(j) *= nSample(j)
            for (j <- 0 until sizeBatch) alpha(idx+j) += delta(j)
            v += xSample * (invLambdaS * delta)
            vz := v + z
            for (j <- 0 until d) w(j) = shrinkage(vz(j), sigma)

            // average
            for (j <- 0 until sizeBatch) abar(idx+j) += delta(j) * invNumLoop
            wbar += w
        }
        wbar *= invNumLoop
        
        (wbar, abar)
    }
    
    
    
    
    /**
     * Minimize the objective w.r.t. w:
     *      (0.5/s)*||X*w - y||_2^2 + lambda*(0.5*||w||_2^2 - w'*z).
     *
     * @param x d-by-s feature matrix
     * @param y s-by-1 label vector
     * @param z d-by-1 vector in the above objective function
     * @param lambda parameter in the above objective function
     * @param normalizers0 (optional) s-by-1 vector whose j-th entry is 1 + ||x_j||_2^2 / 2/ lambda / s
     * @param numEpochs (optional) number of epoches
     * @param alpha0 (optional) initial guess of alpha
     * @param w0 (optional) initial guess of w
     * @return (wbar, abar) the average of the primal-dual variables pair (w, alpha)
     */
    def sdcaQuadraticL2(x: DenseMatrix[Double], y: DenseVector[Double], z: DenseVector[Double], lambda: Double, normalizers0: DenseVector[Double] = null, numEpochs: Int = 20, alpha0: DenseVector[Double] = null, w0: DenseVector[Double] = null): (DenseVector[Double], DenseVector[Double]) = {
        // other parameters
        val sizeBatch: Int = 5
        val isMute = true
        
        // constants
        val s: Int = x.cols
        val d: Int = x.rows
        val invSizeBatch: Double = 1.0 / sizeBatch
        val numLoop: Int = math.floor(s * invSizeBatch).toInt
        val invNumLoop: Double = 1.0 / numLoop.toDouble
        assert(y.length == s)
        val invLambdaS: Double = 1.0 / lambda / s.toDouble
        
        var normalizers: DenseVector[Double] = normalizers0
        if (normalizers0 == null){
            val xColNormsSquare = new Array[Double](s)
            val slambda2: Double = 2.0 * lambda * s
            for (j <- 0 until s) xColNormsSquare(j) = x(::, j).toArray.map(b => b*b).sum
            normalizers = new DenseVector(xColNormsSquare.map(b => 1.0 / (1.0 + b / slambda2)))
        }
        
        // initialization
        val alpha: DenseVector[Double] = if (alpha0 != null) alpha0 else {
            if (w0 != null) (y - x.t * w0)
            else DenseVector.zeros[Double](s)
        }
        val v: DenseVector[Double] = invLambdaS * (x * alpha)
        val ytilde: DenseVector[Double] = y - x.t * z
        
        for (t <- 0 until numEpochs-1) {
            val randIndex: List[Int] = scala.util.Random.shuffle((0 until (s-sizeBatch+1)).toList)
            for (i <- 0 until numLoop) {
                // sampling
                val idx: Int = randIndex(i)
                val r = idx until (idx+sizeBatch)
                val xSample: DenseMatrix[Double] = x(::, r)
                val ySample: DenseVector[Double] = ytilde(r)
                val aSample: DenseVector[Double] = alpha(r)
                val nSample: DenseVector[Double] = normalizers(r) * invSizeBatch
                
                // update
                val delta: DenseVector[Double] = ySample - xSample.t * v - aSample
                for (j <- 0 until sizeBatch) delta(j) *= nSample(j)
                for (j <- 0 until sizeBatch) alpha(idx+j) += delta(j)
                v += xSample * (invLambdaS * delta)
            }
        }
        
        // average options
        var vbar: DenseVector[Double] = DenseVector.zeros[Double](d)
        var abar: DenseVector[Double] = alpha.copy
        val randIndex: List[Int] = scala.util.Random.shuffle((0 until (s-sizeBatch+1)).toList)
        for (i <- 0 until numLoop) {
            // sampling
            val idx: Int = randIndex(i)
            val r = idx until (idx+sizeBatch)
            val xSample: DenseMatrix[Double] = x(::, r)
            val ySample: DenseVector[Double] = ytilde(r)
            val aSample: DenseVector[Double] = alpha(r)
            val nSample: DenseVector[Double] = normalizers(r) * invSizeBatch

            // update
            val delta: DenseVector[Double] = ySample - xSample.t * v - aSample
            for (j <- 0 until sizeBatch) delta(j) *= nSample(j)
            for (j <- 0 until sizeBatch) alpha(idx+j) += delta(j)
            v += xSample * (invLambdaS * delta)

            // average
            for (j <- 0 until sizeBatch) abar(idx+j) += delta(j) * invNumLoop
            vbar += v
        }
        vbar *= invNumLoop
        val wbar: DenseVector[Double] = vbar + z
        
        (wbar, abar)
    }
    

    /**
     * Minimize the objective w.r.t. w:
     *      (0.5/s)*||X*w - y||_2^2 + lambda*(0.5*||w||_2^2 + sigma*||w||_1 - w'*z).
     *
     * @param x d-by-s feature matrix
     * @param y s-by-1 label vector
     * @param z d-by-1 vector in the above objective function
     * @param lambda parameter in the above objective function
     * @param sigma parameter in the above objective function
     * @param maxIterOutter max iteration of the outer loop
     * @param w0 initialization of w
     */
    def acceSdcaQuadratic(x: DenseMatrix[Double], y: DenseVector[Double], z: DenseVector[Double], lambda: Double, sigma: Double, maxIterOutter: Int = 10, w0: DenseVector[Double]): DenseVector[Double] = {
        // parameter
        val isMute: Boolean = true
        val maxIterSdca: Int = 5
        
        // constants
        val s: Int = x.cols
        val d: Int = x.rows
        val xColNormsSquare: Array[Double] = new Array[Double](s)
        for (i <- 0 until s) xColNormsSquare(i) = x(::, i).toArray.map(b => b*b).sum
        val radiusSquare: Double = xColNormsSquare.max
        val kappa: Double = radiusSquare / s.toDouble - lambda
        val mu: Double = lambda / 2
        val rho: Double = mu + kappa
        val eta: Double = math.sqrt(mu / rho)
        val beta: Double = (1.0 - eta) / (1.0 + eta)
        
        val isIllConditioned: Boolean = radiusSquare > 10.0*lambda*s
        if (!isMute) {
            if (!isIllConditioned) println("    Well conditioned! Run Prox-SDCA.")
            else println("    Ill conditioned! Run Accelerated Prox-SDCA.")
        }
        
        if (!isIllConditioned) {
            val slambda2: Double = 2.0 * lambda * s
            val normalizers: DenseVector[Double] = new DenseVector(xColNormsSquare.map(b => 1.0 / (1.0 + b / slambda2)))
            val wa: (DenseVector[Double], DenseVector[Double]) = if (sigma > 1E-10) {
                SdcaL1L2.sdcaQuadraticL1L2(x, y, z, lambda, sigma, normalizers, maxIterOutter*maxIterSdca, null, w0)
            }
            else {
                SdcaL1L2.sdcaQuadraticL2(x, y, z, lambda, normalizers, maxIterOutter*maxIterSdca, null, w0)
            }
            
            return wa._1
        }
        else {
            // initialization       
            val alpha: DenseVector[Double] = y - x.t * w0
            val v: DenseVector[Double] = w0.copy
            val w: DenseVector[Double] = w0.copy
            val wOld: DenseVector[Double] = w.copy
            
            // additional parameters
            val lambda2: Double = lambda + kappa
            val slambda2: Double = 2.0 * lambda2 * s
            val normalizers: DenseVector[Double] = new DenseVector(xColNormsSquare.map(b => 1.0 / (1.0 + b / slambda2)))
            
            for (t <- 0 until maxIterOutter) {
                // for debug purpose
                if (!isMute) {
                    val res: DenseVector[Double] = x.t * w - y
                    val loss: Double = (res.toArray.map(b => b*b).sum) * 0.5 / s.toDouble
                    val wArray: Array[Double] = w.toArray
                    var wz: Double = 0.0
                    for (j <- 0 until d) wz += wArray(j) * z(j)
                    val reg: Double = (wArray.map(b => b*b).sum) * 0.5 + (wArray.map(b => math.abs(b)).sum) - wz
                    val obj: Double = loss + lambda * reg
                    println("    AcceSDCA: objective value is " + obj.toString)
                }
                
                wOld := w
                val wa: (DenseVector[Double], DenseVector[Double]) = if (sigma > 1E-10) {
                    SdcaL1L2.sdcaQuadraticL1L2(x, y, (lambda*z+kappa*v)/lambda2, lambda2, lambda*sigma/lambda2, normalizers, maxIterSdca, alpha, null)
                }
                else {
                    SdcaL1L2.sdcaQuadraticL2(x, y, (lambda*z+kappa*v)/lambda2, lambda2, normalizers, maxIterSdca, alpha, null)
                }
                
                w := wa._1
                alpha := wa._2
                v := (1.0 + beta) * w - beta * wOld
            }
            
            return w
        }
    }
    
}