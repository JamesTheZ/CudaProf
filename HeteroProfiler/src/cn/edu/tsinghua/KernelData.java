package cn.edu.tsinghua;

import java.util.Hashtable;

/**
 * @author ZZ
 * 
 */
public class KernelData {

	private String name;

	private Hashtable<String, Integer> basicInfo;
	private Hashtable<String, Float> occupancy;
	private Hashtable<String, Float> efficiency;
	private Hashtable<String, Float> utilization;
	private Hashtable<String, Float> overhead;
	private Hashtable<String, Long> instruction;
	private Hashtable<String, Float> othersFloat;
	
	private int theoryLimiter, achievedLimiter;

	public KernelData() {
		super();

		basicInfo = new Hashtable<String, Integer>();
		occupancy = new Hashtable<String, Float>();
		efficiency = new Hashtable<String, Float>();
		utilization = new Hashtable<String, Float>();
		overhead = new Hashtable<String, Float>();
		instruction = new Hashtable<String, Long>();
		othersFloat = new Hashtable<String, Float>();

		initHashtables();
	}

	private void initHashtables() {
		basicInfo.put("device", -1);
		basicInfo.put("gridX", -1);
		basicInfo.put("gridY", -1);
		basicInfo.put("gridZ", -1);
		basicInfo.put("blockX", -1);
		basicInfo.put("blockY", -1);
		basicInfo.put("blockZ", -1);
		basicInfo.put("dynamicSharedMem", -1);
		basicInfo.put("staticSharedMem", -1);
		basicInfo.put("localMemPerThread", -1);
		basicInfo.put("localMemTotal", -1);
		basicInfo.put("regPerThread", -1);
		basicInfo.put("cacheConfigReq", -1);
		basicInfo.put("cacheConfigUsed", -1);
		basicInfo.put("sharedMemConfigUsed", -1);
		
		occupancy.put("theory_occupancy", (float) -1.0);
		occupancy.put("achieved_occupancy", (float) -1.0);

		efficiency.put("sm_efficiency", (float) -1.0);
		efficiency.put("gld_efficiency", (float) -1.0);
		efficiency.put("gst_efficiency", (float) -1.0);
		efficiency.put("warp_execution_efficiency", (float) -1.0);
		efficiency.put("warp_nonpred_execution_efficiency", (float) -1.0);
		efficiency.put("shared_efficiency", (float) -1.0);

		utilization.put("dram_utilization", (float) -1.0);
		utilization.put("l1_shared_utilization", (float) -1.0);
		utilization.put("l2_utilization", (float) -1.0);
		utilization.put("ldst_fu_utilization", (float) -1.0);
		utilization.put("alu_fu_utilization", (float) -1.0);

		overhead.put("global_replay_overhead", (float) -1.0);
		overhead.put("local_memory_overhead", (float) -1.0);
		overhead.put("shared_replay_overhead", (float) -1.0);

		instruction.put("gld_throughput", (long) -1);
		instruction.put("gst_throughput", (long) -1);
		instruction.put("dram_read_throughput", (long) -1);
		instruction.put("dram_write_throughput", (long) -1);
		instruction.put("ldst_executed", (long) -1);
		instruction.put("inst_fp_32", (long) -1);
		instruction.put("inst_fp_64", (long) -1);
		instruction.put("inst_integer", (long) -1);
		instruction.put("shared_load_replay", (long) -1);
		instruction.put("shared_store_replay", (long) -1);
		instruction.put("l2_total_misses", (long) -1);
		instruction.put("l2_total_hit", (long) -1);
		instruction.put("l1_global_load_miss", (long) -1);
		instruction.put("l1_global_load_hit", (long) -1);
		instruction.put("inst_executed", (long) -1);
		instruction.put("shared_load", (long) -1);
		instruction.put("shared_store", (long) -1);
		instruction.put("local_load", (long) -1);
		instruction.put("local_store", (long) -1);
		instruction.put("gld_request", (long) -1);
		instruction.put("gst_request", (long) -1);

		othersFloat.put("ipc", (float) -1.0);
		othersFloat.put("l1_cache_global_hit_rate", (float) -1.0);
	}

	/**
	 * @return the name
	 */
	public String getName() {
		return name;
	}

	/**
	 * @param name
	 *            the name to set
	 */
	public void setName(String name) {
		this.name = name;
	}

	public void addBasicInfoItem(String key, int value) {
		basicInfo.put(key, value);
	}

	public void addOccupancyItem(String key, float value) {
		occupancy.put(key, value);
	}

	public void addEfficiencyItem(String key, float value) {
		efficiency.put(key, value);
	}

	public void addUtilizationItem(String key, float value) {
		utilization.put(key, value);
	}

	public void addOverheadItem(String key, float value) {
		overhead.put(key, value);
	}

	public void addInstructionItem(String key, Long value) {
		instruction.put(key, value);
	}

	public void addOthersItem(String key, float value) {
		othersFloat.put(key, value);
	}

	public Hashtable<String, Integer> getBasicInfo() {
		return basicInfo;
	}

	public Hashtable<String, Float> getOccupancy() {
		return occupancy;
	}

	public Hashtable<String, Float> getEfficiency() {
		return efficiency;
	}

	public Hashtable<String, Float> getUtilization() {
		return utilization;
	}

	public Hashtable<String, Float> getOverhead() {
		return overhead;
	}

	public Hashtable<String, Long> getInstruction() {
		return instruction;
	}

	public Hashtable<String, Float> getOthers() {
		return othersFloat;
	}

	public int getTheoryLimiter() {
		return theoryLimiter;
	}

	public void setTheoryLimiter(int theoryLimiter) {
		this.theoryLimiter = theoryLimiter;
	}

	public int getAchievedLimiter() {
		return achievedLimiter;
	}

	public void setAchievedLimiter(int achievedLimiter) {
		this.achievedLimiter = achievedLimiter;
	}

}
