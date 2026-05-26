ThisBuild / version      := "0.1.0"
ThisBuild / scalaVersion := "2.12.18"

name := "microservice_3"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.5.0",
  "org.apache.spark" %% "spark-sql"  % "3.5.0"
)
