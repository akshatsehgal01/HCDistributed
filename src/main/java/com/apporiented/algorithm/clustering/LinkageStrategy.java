package com.apporiented.algorithm.clustering;

import java.util.Collection;

public interface LinkageStrategy {

	public Distance calculateDistance(Collection<Distance> distances);
}
