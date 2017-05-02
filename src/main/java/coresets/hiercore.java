package coresets;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import weka.core.converters.CSVLoader;

import weka.clusterers.HierarchicalClusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class hiercore {
	
	public static Instances readDataFile(String filename) throws IOException {
		//Hello World
		System.out.println("---------------Reading data file------------------");
		BufferedReader inputReader = null;
		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}
		Instances data = new Instances(inputReader);
		return data;
	}
	
	public static Instances BuildCoreset(String InputFile, double NeighbourPartPerc,Instances data) throws Exception{
		System.out.println("-------------Building the coresets----------------");
		int NeighbourHoodPartition = (int) Math.ceil(NeighbourPartPerc/100*data.numInstances());
		System.out.println(NeighbourHoodPartition);
		SimpleKMeans kmeans = new SimpleKMeans();
		kmeans.setSeed(10);
		//important parameter to set: preserver order, number of cluster.
		kmeans.setPreserveInstancesOrder(true);
		kmeans.setNumClusters(NeighbourHoodPartition);
		kmeans.buildClusterer(data);
		// This array returns the cluster number (starting with 0) for each instance
		// The array has as many elements as the number of instances
		int[] assignments = kmeans.getAssignments();
		int i=0;
		System.out.println("#" + "    " + "--------------Instances-----------------------");
		for(int clusterNum : assignments) {
			System.out.println(clusterNum + " => " + data.get(i));
		    i++;
		}
		Instances CoreInstance = kmeans.getClusterCentroids();
		System.out.println("---------------------------------------------------");
		return CoreInstance;
	}
	
	public static void main(String[] args) throws Exception
	{
		String InputFile = "/home/akshat/workspace/HCDistributed/src/main/java/trainingdata/diabetes.arff";
		double NeighbourPartPerc = 100;
		Instances data = readDataFile(InputFile);
		Instances CoreInstances = BuildCoreset(InputFile,NeighbourPartPerc,data);
		System.out.println(CoreInstances);
		System.out.println("---------------------------------------------------");
		HierarchicalClusterer hc = new HierarchicalClusterer(true);
		hc.buildClusterer(data);
		System.out.println("Euclidean Original DataSet: " + hc.graph());
		System.out.println("---------------------------------------------------");
		hc.buildClusterer(CoreInstances);
		System.out.println("Euclidean Coresets Dataset: " + hc.graph());
		System.out.println("---------------------------------------------------");
		//Manhattan - true
		//Euclidean - false
		//--------------------------------------------------------------
//		HierarchicalClusterer hc = new HierarchicalClusterer(true);
//		hc.buildClusterer(data);
//		System.out.println("Manhattan: " + hc.graph());
		//--------------------------------------------------------------
		
	}
}