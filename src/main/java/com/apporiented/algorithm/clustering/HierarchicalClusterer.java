package com.apporiented.algorithm.clustering;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class HierarchicalClusterer {
	private static List<String> datarows = new ArrayList<String>();

	public HierarchicalClusterer(List<String> dataRows) throws IOException, InterruptedException {
		datarows = dataRows;
	}

	public static class Coord {
		private double[] x;

		public Coord(double[] x) {
			this.x = x;
		}
	}

	private static List<Coord> readCoordinates() {
		List<Coord> coordList = new ArrayList<Coord>();
		Iterator<String> iter = datarows.iterator();

		while (iter.hasNext()) {
			String[] elems = iter.next().split(",");
			double[] x = new double[elems.length];

			try {
				for (int i = 0; i < x.length; i++) {
					x[i] = Integer.parseInt(elems[i]);
				}
			} catch (Exception e) {
				continue;
			}
			coordList.add(new Coord(x));
		}
		return coordList;
	}

	private static Cluster runClusterer() {
		List<Coord> coords = readCoordinates();
		double[][] distances = new double[coords.size()][coords.size()];
		String[] names = new String[coords.size()];
		for (int row = 0; row < coords.size(); row++) {
			Coord coord1 = coords.get(row);
			for (int col = row + 1; col < coords.size(); col++) {
				Coord coord2 = coords.get(col);
				double diff_square_sum = 0.0;
				for (int i = 0; i < coord2.x.length; i++) {
					diff_square_sum += Math.pow(coord2.x[i] - coord1.x[i], 2);
				}
				distances[row][col] = Math.sqrt(diff_square_sum);
				distances[col][row] = Math.sqrt(diff_square_sum);
				System.out.println(diff_square_sum);
			}
			names[row] = "" + row;
		}
		ClusteringAlgorithm alg = new DefaultClusteringAlgorithm();
		Cluster cluster = alg.performClustering(distances, names, new SingleLinkageStrategy());
		return cluster;
	}

	public String toHCString() {
		Cluster cluster = runClusterer();
		return cluster.toNewick();
	}
}
