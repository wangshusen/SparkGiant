package distopt.utils

// others
import scala.math
// breeze
import breeze.linalg._
import breeze.numerics._


class CG(d: Int) {
    val r: DenseVector[Double] = DenseVector.zeros[Double](d)
    val p: DenseVector[Double] = DenseVector.zeros[Double](d)
    val ap: DenseVector[Double] = DenseVector.zeros[Double](d)
    var cgTol: Double = 1E-20
    
    
    
    /**
     * Solve (A * A' + lam * I) * w = b.
     *
     * @param a d-by-s dense matrix
     * @param b d-by-1 dense matrix
     * @param lam regularization parameter
     * @param maxiter max number of iterations
     * @return w the solution
     */
    def solver1(a: DenseMatrix[Double], b: DenseVector[Double], lam: Double, maxiter: Int = 20): Array[Double] = {
        val tol: Double = this.cgTol * math.sqrt(b.toArray.map(x => x*x).sum)
        val w: DenseVector[Double] = DenseVector.zeros[Double](this.d)
        this.r := b //- lam * w - a * (a.t * w)
        this.p := r
        var rsold: Double = this.r.toArray.map(x => x*x).sum
        var rsnew: Double = 0.0
        var alpha: Double = 0.0
        var rssqrt: Double = 0.0
        
        for (q <- 0 until maxiter) {
            this.ap := lam * this.p + a * (a.t * this.p)
            var pap: Double = 0.0
            for (j <- 0 until d) pap += this.p(j) * this.ap(j)
            alpha = rsold / pap
            w += alpha * this.p
            this.r -= alpha * this.ap
            rsnew = this.r.toArray.map(a => a*a).sum
            rssqrt = math.sqrt(rsnew)
            if (rssqrt < tol) {
                //println("Iter " + q.toString + ": converged! res = " + rssqrt.toString)
                return w.toArray
            }
            this.p *= rsnew / rsold
            this.p += this.r
            rsold = rsnew
        }
        this.cgTol *= 0.5
        w.toArray
    }
    
    
    /**
     * Solve (H + lam * I) * w = b.
     *
     * @param h d-by-d dense matrix
     * @param b d-by-1 dense matrix
     * @param lam regularization parameter
     * @param maxiter max number of iterations
     * @return w the solution
     */
    def solver2(h: DenseMatrix[Double], b: DenseVector[Double], lam: Double, maxiter: Int = 20): Array[Double] = {
        val d: Int = h.rows
        val tol: Double = this.cgTol * math.sqrt(b.toArray.map(x => x*x).sum)
        val w: DenseVector[Double] = DenseVector.zeros[Double](this.d)
        this.r := b //- lam * w - h * w
        this.p := this.r
        var rsold: Double = r.toArray.map(a => a*a).sum
        var rsnew: Double = 0.0
        var alpha: Double = 0.0
        var rssqrt: Double = 0.0
        
        for (j <- 0 until this.d) h(j, j) += lam
        
        for (q <- 0 until maxiter) {
            this.ap := h * this.p
            var pap: Double = 0.0
            for (j <- 0 until d) pap += this.p(j) * this.ap(j)
            alpha = rsold / pap
            w += alpha * this.p
            this.r -= alpha * this.ap
            rsnew = this.r.toArray.map(x => x*x).sum
            rssqrt = math.sqrt(rsnew)
            if (rssqrt < tol) {
                //println("Iter " + q.toString + ": converged! res = " + rssqrt.toString)
                return w.toArray
            }
            this.p *= rsnew / rsold
            this.p += this.r
            rsold = rsnew
        }
        this.cgTol *= 0.5
        w.toArray
    }
    
}