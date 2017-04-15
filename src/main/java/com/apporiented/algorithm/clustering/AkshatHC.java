package com.apporiented.algorithm.clustering;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class AkshatHC {

	public static class TokenizerMapper extends Mapper<Object, Text, Text, Text> {

		public static int numPartitions;
		private static final Text HC = new Text("1");

		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			Random r = new Random();
			int partition = r.nextInt(3);
			HC.set(Integer.toString(partition));
			context.write(HC, value);
		}
	}

	public static class IntSumReducer extends Reducer<Text, Text, Text, Text> {
		private static final Text Data = new Text();
		public void reduce(Text key, Iterable <Text> value, Context context) throws IOException, InterruptedException {
			List<String> datarows = new ArrayList<String>();
			Iterator<Text> iter = value.iterator();
			while(iter.hasNext()){
				String temp = iter.next().toString();
				datarows.add(new String(temp));
			}
			HierarchicalClusterer JMD = new HierarchicalClusterer(datarows);
			Data.set(JMD.toHCString());
			context.write(key, Data);
		}
	}

	public static void main(String[] args) throws Exception {

		double startTime = (double) System.currentTimeMillis();
		Configuration conf = new Configuration();
		String inputPath = args[0];
		String outputPath = args[1];

		// number of partitions for PARABLE
		TokenizerMapper.numPartitions = Integer.parseInt(args[2]);

		if (args.length < 3) {
			System.err.println("AkshatHC usage: [input-path] [num-reducers]");
			System.exit(0);
		}

		Job job = Job.getInstance(conf, "AkshatHC");
		job.setJobName(AkshatHC.class.getSimpleName());
		job.setJarByClass(AkshatHC.class);
		FileInputFormat.setInputPaths(job, new Path(inputPath));
		FileOutputFormat.setOutputPath(job, new Path(outputPath));

		job.setMapperClass(TokenizerMapper.class);
		job.setReducerClass(IntSumReducer.class);

		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);

		if (job.waitForCompletion(true)) {
			double endTime = System.currentTimeMillis();
			System.out.println("Total time taken by the map reduce job: " + (endTime - startTime) / 1000 + " seconds");
			System.exit(0);
		} else {
			System.exit(1);
		}
	}
}