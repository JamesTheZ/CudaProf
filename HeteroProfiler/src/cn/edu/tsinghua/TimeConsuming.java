/**
 * 
 */
package cn.edu.tsinghua;

import java.util.ArrayList;
import java.util.Hashtable;

/**
 * @author zhengzhen
 *
 */
public class TimeConsuming {
	
	private Hashtable<String, Long> kernelTimeCons;
	private Hashtable<String, Long> memcpyTimeCons;
	private long cpuTimeCons;
	
	public TimeConsuming() {
		super();
		kernelTimeCons = new Hashtable<String, Long>();
		memcpyTimeCons = new Hashtable<String, Long>();
	}
	
	public void addKerTimeCons(String kerName, long timeCons) {
		kernelTimeCons.put(kerName, timeCons);
	}
	
	public void addCpyTimeCons(String cpyStyle, long timeCons) {
		memcpyTimeCons.put(cpyStyle, timeCons);
	}
	
	public void setCpuTimeCons(long cpuCons) {
		cpuTimeCons = cpuCons;
	}
	
	public Hashtable<String, Long> getKerTimeCons() {
		return kernelTimeCons;
	}
	
	public Hashtable<String, Long> getCpyTimeCons() {
		return memcpyTimeCons;
	}
	
	public long getCpuTimeCons() {
		return cpuTimeCons;
	}
	
	
}
