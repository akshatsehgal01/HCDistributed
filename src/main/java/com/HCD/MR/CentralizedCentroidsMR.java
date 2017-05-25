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

public class CentralizedCentroidsMR {

	public static class TokenizerMapper extends Mapper<Object, Text, Text, Text> {
		private static final Text HC = new Text("1");

		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			HC.set("1");
			context.write(HC, value);
		}
	}

	public static class IntSumReducer extends Reducer<Text, Text, Text, Text> {
		public void reduce(Text key, Iterable<Text> value, Context context) throws IOException, InterruptedException {
			Configuration conf = context.getConfiguration();
			FileSystem fs = FileSystem.get(conf);
			Path attributesFile = new Path(conf.get("attributePath"));
			Path fullDataFile = new Path(conf.get("fullDataPath"));
			String attribute = new Scanner(fs.open(attributesFile)).useDelimiter("\\Z").next();
			String content = new Scanner(fs.open(fullDataFile)).useDelimiter("\\Z").next();
			content = attribute +"\n"+ content;
			BufferedReader br_actual = new BufferedReader(new StringReader(content));
			Instances actual = new Instances(br_actual);
			HierarchicalClusterer HC = new HierarchicalClusterer(true, Integer.parseInt(conf.get("noOfClusters")), attribute);
			try{
				HC.buildClusterer(actual);
				context.write(key, new Text(HC.graph()));
			}catch(Exception e)
			{
				context.write(key, new Text("lag gaye lode"));
			}
		}
	}

	public static void main(String[] args) throws Exception {

		double startTime = System.currentTimeMillis();
		Configuration conf = new Configuration();
		// for google cloud
		conf.set("fs.default.name", "gs://mihir");
		String dummyInputPath = args[0];
		String attributePath = args[1];
		String fullDataPath = args[2];
		String dummyOutputPath = args[3];
		String noOfClusters = args[4];
		conf.set("fullDataPath", fullDataPath);
		conf.set("attributePath", attributePath);
		conf.set("noOfClusters", noOfClusters);
		Job job = Job.getInstance(conf, "CentralizedCentroidsMR");
		job.setJobName(CentralizedCentroidsMR.class.getSimpleName());
		job.setJarByClass(CentralizedCentroidsMR.class);
		FileInputFormat.setInputPaths(job, new Path(dummyInputPath));
		FileOutputFormat.setOutputPath(job, new Path(dummyOutputPath));

		job.setMapperClass(TokenizerMapper.class);
		job.setReducerClass(IntSumReducer.class);

		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);

		if (job.waitForCompletion(true)) {
			double endTime = System.currentTimeMillis();
			System.out.println("Total training time: " + (endTime - startTime) / 1000 + " seconds");
		} else {
			System.exit(1);
		}
	}
}