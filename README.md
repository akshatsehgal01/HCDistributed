# HCDistributed
Hierarchical Clustering Approximation Algorithm implementation on MapReduce framework.

#SETUP INSTRUCTIONS
This project uses Hadoop 2.7.2 and JDK 1.7
Please use the same folder structure while compiling

#BUILDING JAR
We use the Apache Maven Shade Plugin (maven-shade-plugin) to make our jar.
Right Click Project -> Maven -> Add Plugin

To build the jar, right click project -> run as -> maven build -> put goal as "package shade:shade"

#JAR USAGE
hadoop jar "local jar path" "map reduce class file" "hdfs path of input file" "hdfs path of attribute file" "hdfs path of centroid dir" "hdfs path for output dir" "number of partitions" "coreset percentage" "number of clusters"

sample command:
hadoop jar /home/akshat/workspace/HCDistributed/target/HCDistributed-0.0.1-SNAPSHOT.jar com.HCD.MR.AkshatHC2 /input/Akshat/inputfiles/syndata.csv /input/Akshat/inputfiles/Attribute.txt /input/Akshat/Centroids/ /input/Akshat/outputsyndatachck/ 1 100.0 2

NOTE:
input file is a typical CSV file
attribute file contains the header of a typical arff file
please refer to sample input files