package com.apporiented.algorithm.clustering;

import com.apporiented.algorithm.clustering.visualization.*;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import java.awt.BorderLayout;
import java.awt.Color;
import java.io.IOException;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.WindowConstants;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class HCDistributed {

	public static class Coord {
		private double[] x;

		public Coord(double[] x) {
			this.x = x;
		}
	}

	private static Cluster runClusterer(List<Coord> coords) throws IOException {
		double[][] distances = new double[coords.size()][coords.size()];
		String[] names = new String[coords.size()];
		for (int row = 0; row < coords.size(); row++) {
			Coord coord1 = coords.get(row);
			for (int col = row + 1; col < coords.size(); col++) {
				Coord coord2 = coords.get(col);
				double diff_square_sum = 0.0;
				for (int i = 0; i < coord2.x.length; i++) {
					diff_square_sum += Math.sqrt(Math.pow(coord2.x[i] - coord1.x[i], 2));
				}
				distances[row][col] = diff_square_sum;
				distances[col][row] = diff_square_sum;
			}
			names[row] = "" + row;
		}
		ClusteringAlgorithm alg = new DefaultClusteringAlgorithm();
		Cluster cluster = alg.performClustering(distances, names, new SingleLinkageStrategy());
		return cluster;
	}

	public static class TokenizerMapper extends Mapper<Object, Text, Text, Text> {

		public static int numPartitions;
		private static final Text HC = new Text();
		private static final Text Data = new Text();

		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {

			String dataRows[] = value.toString().trim().split("\\r?\\n");

			for (int i = 0; i < dataRows.length; i++) {
				Random r = new Random();
				int partition = r.nextInt(3);
				HC.set(Integer.toString(partition));
				Data.set(dataRows[i]);
				context.write(HC, Data);
			}
		}
	}

	public static class IntSumReducer extends Reducer<Text, Text, Text, Text> {
		private Text data = new Text();

		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			List<Coord> coords = new ArrayList<Coord>();
			Iterator<Text> iter = values.iterator();
			while (iter.hasNext()) {
				String[] elems = iter.next().toString().trim().split(",");
				double[] x = new double[elems.length];
				try {
					for (int i = 0; i < x.length; i++) {
						x[i] = Double.parseDouble(elems[i]);
					}
				} catch (Exception e) {
					continue;
				}
				coords.add(new Coord(x));
			}
			try {
				Cluster cluster = runClusterer(coords);
				data.set(cluster.toNewick()+";");
				context.write(key, data);
			} catch (IOException e) {
				e.printStackTrace();
			}
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
			System.err.println("HCDistributed usage: [input-path] [num-reducers]");
			System.exit(0);
		}

		Job job = Job.getInstance(conf, "HCDistributed");
		job.setJobName(HCDistributed.class.getSimpleName());
		job.setJarByClass(HCDistributed.class);
		FileInputFormat.setInputPaths(job, new Path(inputPath));
		FileOutputFormat.setOutputPath(job, new Path(outputPath));

		job.setMapperClass(TokenizerMapper.class);
		job.setCombinerClass(IntSumReducer.class);
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