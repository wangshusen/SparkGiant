name := "giant"

version := "1.0"

scalaVersion := "2.11.8"

libraryDependencies ++= Seq( "org.apache.spark" %% "spark-core" % "2.1.1",
                            "org.apache.spark" %% "spark-sql" % "2.1.1",
                            "org.apache.spark" %% "spark-mllib" % "2.1.1" % "provided",
                            "com.github.fommil.netlib" % "all" % "1.1.2" pomOnly()
                           )
resolvers += "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"

