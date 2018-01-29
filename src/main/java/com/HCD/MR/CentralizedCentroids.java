package com.HCD.MR;

import java.util.Scanner;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
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
import weka.core.Instances;

public class CentralizedCentroids {

	public static class TokenizerMapper extends Mapper<Object, Text, Text, Text> {
		private static final Text HC = new Text("1");

		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			HC.set("1");
			context.write(HC, value);
		}
	}

	public static class IntSumReducer extends Reducer<Text, Text, Text, Text> {
		public void reduce(Text key, Iterable<Text> value, Context context) throws IOException, InterruptedException {
			context.write(key, new Text("check"));
		}
	}

	public static void main(String[] args) throws Exception {

		Configuration conf = new Configuration();
		// for google cloud
		//conf.set("fs.default.name", "gs://hcd1");
		String dummyInputPath = args[0];
		String attributePath = args[1];
		String fullDataPath = args[2];
		String centralizedCentroidsPath = args[3];
		String dummyOutputPath = args[4];
		String noOfClusters = args[5];
		Job job = Job.getInstance(conf, "CentralizedCentroids");
		job.setJobName(CentralizedCentroids.class.getSimpleName());
		job.setJarByClass(CentralizedCentroids.class);
		FileInputFormat.setInputPaths(job, new Path(dummyInputPath));
		FileOutputFormat.setOutputPath(job, new Path(dummyOutputPath));

		job.setMapperClass(TokenizerMapper.class);
		job.setReducerClass(IntSumReducer.class);

		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);

		if (job.waitForCompletion(true)) {
			double startTime = System.currentTimeMillis();
			Path attributesFile = new Path(attributePath);
			Path fullDataFile = new Path(fullDataPath);
			Path centralizedCentroidsFile = new Path(centralizedCentroidsPath + "centroids");
			FileSystem fs = FileSystem.get(conf);
			String attribute = new Scanner(fs.open(attributesFile)).useDelimiter("\\Z").next();
			String content = new Scanner(fs.open(fullDataFile)).useDelimiter("\\Z").next();
			content = attribute +"\n"+ content;
			BufferedReader br_actual = new BufferedReader(new StringReader(content));
			Instances actual = new Instances(br_actual);
			HierarchicalClusterer HC = new HierarchicalClusterer();
			HC.buildClusterer(actual);
			BufferedWriter br = new BufferedWriter(new OutputStreamWriter(fs.create(centralizedCentroidsFile, true)));
			br.write(HC.graph());
			br.close();
			double endTime = System.currentTimeMillis();
			System.out.println("Total training time: " + (endTime - startTime) / 1000 + " seconds");
		} else {
			System.exit(1);
		}
	}
}