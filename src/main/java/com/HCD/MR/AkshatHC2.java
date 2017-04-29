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
            // HC.set(Integer.toString(1));
            context.write(HC, value);
        }
    }

    public static class IntSumReducer extends Reducer<Text, Text, Text, Text> {
        private static final Text Data = new Text();

        public void reduce(Text key, Iterable<Text> value, Context context) throws IOException, InterruptedException {
            Configuration conf = context.getConfiguration();
            String attrPath = conf.get("Path");
            //double pp = Double.parseDouble(conf.get("partitionPercentage"));

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
            HierarchicalClusterer HC = new HierarchicalClusterer();
            String actualNewick = "";
            String approxNewick = "";
            try {
                HC.buildClusterer(data);
                actualNewick = HC.graph();
                Instances CoreInstances = BuildCoreset(80,data);
                HC.buildClusterer(CoreInstances);
                approxNewick = HC.graph();
            } catch (Exception e) {
            }
            String newick = "actualNewick:"+actualNewick+"\n"+"approxNewick:"+approxNewick;
            context.write(key, new Text(newick));
        }
        
        public static Instances BuildCoreset(double NeighbourPartPerc,Instances data) throws Exception{
    		//System.out.println("-------------Building the coresets----------------");
    		int NeighbourHoodPartition = (int) Math.ceil(NeighbourPartPerc/100*data.numInstances());
    		//System.out.println(NeighbourHoodPartition);
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
    		//System.out.println("#" + "    " + "--------------Instances-----------------------");
    		/*for(int clusterNum : assignments) {
    			System.out.println(clusterNum + " => " + data.get(i));
    		    i++;
    		}*/
    		Instances CoreInstance = kmeans.getClusterCentroids();
    		//System.out.println("---------------------------------------------------");
    		return CoreInstance;
    	}
    }

    public static void main(String[] args) throws Exception {

        double startTime = (double) System.currentTimeMillis();
        Configuration conf = new Configuration();
        String inputPath = args[0];
        String attributePath = args[1];
        String outputPath = args[2];
        String partitions = args[3];
        //String partitionPercentage = args[4];

        if (args.length < 4) {
            System.err.println("AkshatHC2 usage: [input-path] [num-reducers]");
            System.exit(0);
        }
        conf.set("Path", attributePath);
        conf.set("Partitions", partitions);
        //conf.set("PartitionPercentage", partitionPercentage);
        Job job = Job.getInstance(conf, "AkshatHC2");
        job.setJobName(AkshatHC2.class.getSimpleName());
        job.setJarByClass(AkshatHC2.class);
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