package com.HCD.MR;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.StringReader;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import weka.clusterers.HierarchicalClusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class AkshatHC2 {

	public static boolean gotAttributes = false;
	public static String attributes = "";

	public static int x;

	public static class TokenizerMapper extends Mapper<Object, Text, Text, Text> {

		public static int numPartitions;
		private static final Text HC = new Text("1");

		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			Configuration conf = context.getConfiguration();
			String param = conf.get("Partitions");
			Random r = new Random();
			int partition = r.nextInt(Integer.valueOf(param));
			HC.set(Integer.toString(partition));
			context.write(HC, value);
		}
	}

	public static class IntSumReducer extends Reducer<Text, Text, Text, Text> {
		private static final Text Data = new Text();

		public void reduce(Text key, Iterable<Text> value, Context context) throws IOException, InterruptedException {
			Configuration conf = context.getConfiguration();
			String attrPath = conf.get("Path");
			String param = conf.get("Partitions");
			int partition = Integer.valueOf(param);
			double pp = Double.parseDouble(conf.get("PartitionPercentage"));
			int noOfClusters = Integer.parseInt(conf.get("NoOfClusters"));
			boolean getApprox = true;
			if (pp == 100.0 || (partition == 1 && pp == 100.0)) {
				getApprox = false;
			}

			if (!gotAttributes) {
				try {
					Path pt = new Path(attrPath);
					FileSystem fs = FileSystem.get(conf);
					BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(pt)));
					String line;
					line = br.readLine();
					while (line != null) {
						attributes = attributes + line + "\n";
						line = br.readLine();
					}
				} catch (Exception e) {
					attributes = "exception";
				}
				gotAttributes = true;
			}
			Iterator<Text> iter = value.iterator();
			String fullArff = "";
			while (iter.hasNext()) {
				fullArff += iter.next().toString() + "\n";
			}
			fullArff = fullArff.trim();
			fullArff = attributes + fullArff;
			BufferedReader br = new BufferedReader(new StringReader(fullArff));
			Instances data = new Instances(br);
			HierarchicalClusterer HC = new HierarchicalClusterer(true, noOfClusters, attributes);
			String actualCentroids = "";
			String approxCentroids = "";
			try {
				HC.buildClusterer(data);
				actualCentroids = HC.graph();
				if (getApprox) {
					Instances CoreInstances = BuildCoreset(pp, data);
					HC.buildClusterer(CoreInstances);
					approxCentroids = HC.graph();
					approxCentroids = approxCentroids.replace('#', '%');
				} else {
					approxCentroids = "Coresets disabled!\n";
				}
			} catch (Exception e) {
				approxCentroids = "approximation too small!\n";
			}
			String centroids = "actualCentroids:" + actualCentroids + "\n" + "approxCentroids:" + approxCentroids;
			context.write(key, new Text(centroids));
		}

		public static Instances BuildCoreset(double NeighbourPartPerc, Instances data) throws Exception {
			// System.out.println("-------------Building the
			// coresets----------------");
			int NeighbourHoodPartition = (int) Math.ceil(NeighbourPartPerc / 100 * data.numInstances());
			// System.out.println(NeighbourHoodPartition);
			SimpleKMeans kmeans = new SimpleKMeans();
			kmeans.setSeed(10);
			// important parameter to set: preserver order, number of cluster.
			kmeans.setPreserveInstancesOrder(true);
			kmeans.setNumClusters(NeighbourHoodPartition);
			kmeans.buildClusterer(data);
			// This array returns the cluster number (starting with 0) for each
			// instance
			// The array has as many elements as the number of instances
			int[] assignments = kmeans.getAssignments();
			int i = 0;
			// System.out.println("#" + " " +
			// "--------------Instances-----------------------");
			String chk = "";
			/*
			 * for(int clusterNum : assignments) {
			 * //System.out.println(clusterNum + " => " + data.get(i)); chk =
			 * chk+clusterNum + " => " + data.get(i)+"\n"; i++; }
			 */
			// context.write(new Text("1"), new Text(chk));
			Instances CoreInstance = kmeans.getClusterCentroids();
			// System.out.println("---------------------------------------------------");
			return CoreInstance;
		}
	}

	public static void main(String[] args) throws Exception {

		double startTime = System.currentTimeMillis();
		Configuration conf = new Configuration();
		String trainingPath = args[0];
		String attributePath = args[1];
		String outputPath = args[2];
		String partitions = args[3];
		String partitionPercentage = args[4];
		String noOfClusters = args[5];
		conf.set("Path", attributePath);
		conf.set("Partitions", partitions);
		conf.set("PartitionPercentage", partitionPercentage);
		conf.set("NoOfClusters", noOfClusters);
		Job job = Job.getInstance(conf, "AkshatHC2");
		job.setJobName(AkshatHC2.class.getSimpleName());
		job.setJarByClass(AkshatHC2.class);
		FileInputFormat.setInputPaths(job, new Path(trainingPath));
		FileOutputFormat.setOutputPath(job, new Path(outputPath));

		job.setMapperClass(TokenizerMapper.class);
		job.setReducerClass(IntSumReducer.class);

		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);

		if (job.waitForCompletion(true)) {
			Path pt = new Path(outputPath + "/part-r-00000");
			FileSystem fs = FileSystem.get(conf);
			BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(pt)));
			String ApproxCentroid = "";
			String ActualCentroid = "";
			try {
				StringBuilder sb = new StringBuilder();
				String line = br.readLine();
				while (line != null) {
					if (line.contains("#")) {
						line = line.substring(line.indexOf("#") + 1);
						line = line.substring(0, line.indexOf("#"));
						ActualCentroid = ActualCentroid + line + "\n";
					}
					if (line.contains("%")) {
						line = line.substring(line.indexOf("%") + 1);
						line = line.substring(0, line.indexOf("%"));
						ApproxCentroid = ApproxCentroid + line + "\n";
					}
					sb.append(line);
					sb.append(System.lineSeparator());
					line = br.readLine();
				}
			} finally {
				br.close();
			}
			/*
			 * System.out.println("-------APPROX--------");
			 * System.out.println(ApproxCentroid);
			 * System.out.println("-------ACTUAL--------");
			 * System.out.println(ActualCentroid);
			 */
			String attr = "";
			Path attr_pt = new Path(attributePath);
			BufferedReader br_attr = new BufferedReader(new InputStreamReader(fs.open(attr_pt)));
			String line;
			line = br_attr.readLine();
			while (line != null) {
				attr = attr + line + "\n";
				line = br_attr.readLine();
			}
			SimpleKMeans kmeans = new SimpleKMeans();
			kmeans.setSeed(10);
			kmeans.setPreserveInstancesOrder(true);
			kmeans.setNumClusters(Integer.parseInt(noOfClusters));
			ActualCentroid = attr + ActualCentroid;
			BufferedReader br_actual = new BufferedReader(new StringReader(ActualCentroid));
			Instances actualCentroidData = new Instances(br_actual);
			kmeans.buildClusterer(actualCentroidData);
			Instances actualCentroidInstance = kmeans.getClusterCentroids();
			if (!(Double.parseDouble(partitionPercentage) == 100.0
					|| (Integer.parseInt(partitions) == 1 && Double.parseDouble(partitionPercentage) == 100.0))) {
				BufferedReader br_approx = new BufferedReader(new StringReader(ApproxCentroid));
				ApproxCentroid = attr + ApproxCentroid;
				Instances approxCentroidData = new Instances(br_approx);
				kmeans.buildClusterer(approxCentroidData);
				Instances approxCentroidInstance = kmeans.getClusterCentroids();
				System.out.println("-------APPROX--------");
				System.out.println(approxCentroidInstance);
				System.out.println("-------ACTUAL--------");
				System.out.println(actualCentroidInstance);
			} else {
				System.out.println("-------ACTUAL--------");
				System.out.println(actualCentroidInstance);
				System.exit(0);
			}
			double endTime = System.currentTimeMillis();
			System.out.println("Total training time: " + (endTime - startTime) / 1000 + " seconds");
		} else {
			System.exit(1);
		}
	}
}