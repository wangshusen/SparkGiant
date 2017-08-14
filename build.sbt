name := "giant"

version := "1.0"

scalaVersion := "2.11.5"

libraryDependencies ++= Seq( "org.apache.spark" %% "spark-core" % "2.2.0" % "provided",
                        "org.apache.spark" %% "spark-sql" % "2.2.0" % "provided",
                        "org.apache.spark" %% "spark-mllib" % "2.2.0" % "provided",
                        "com.github.fommil.netlib" % "all" % "1.1.2" % "provided",
                        "org.scalanlp" %% "breeze" % "0.12",
                        "org.scalanlp" %% "breeze-viz" % "0.12"
                       )

resolvers += "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
