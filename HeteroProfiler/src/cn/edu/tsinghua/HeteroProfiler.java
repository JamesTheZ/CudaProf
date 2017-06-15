package cn.edu.tsinghua;

import java.awt.Frame;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.StringTokenizer;

import org.eclipse.swt.widgets.Display;
import org.eclipse.swt.widgets.Shell;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.SWT;
import org.eclipse.swt.widgets.Button;
import org.eclipse.swt.widgets.Composite;
import org.eclipse.swt.widgets.Event;
import org.eclipse.swt.widgets.TabFolder;
import org.eclipse.swt.widgets.TabItem;
import org.eclipse.swt.widgets.Table;
import org.eclipse.swt.widgets.TableColumn;
import org.eclipse.swt.widgets.TableItem;
import org.eclipse.swt.widgets.Text;
import org.eclipse.swt.widgets.FileDialog;
import org.eclipse.swt.widgets.Listener;
import org.eclipse.swt.widgets.Menu;
import org.eclipse.swt.widgets.MenuItem;
import org.eclipse.swt.widgets.Tree;
import org.eclipse.swt.widgets.TreeItem;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.awt.SWT_AWT;
import org.eclipse.swt.events.SelectionAdapter;
import org.eclipse.swt.events.SelectionEvent;
import org.eclipse.swt.graphics.Color;
import org.eclipse.swt.graphics.Font;
import org.eclipse.swt.graphics.Point;
import org.eclipse.swt.graphics.Rectangle;
import org.eclipse.swt.widgets.Label;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.data.general.DefaultPieDataset;

/**
 * @author zhengzhen
 * 
 */
public class HeteroProfiler {

	private Shell shell;
	private Menu menuBar;
	private Text exeCmdText;
	private Button profBtnButton;
	private Button debugButton;
	private Tree tree;
	private TabFolder resultShowTab;
	private TabItem sheetTabItem;
	private TabItem chartTabItem;
	private Text txtLabel;

	private ArrayList<KernelData> kernelDatas;
	private TimeConsuming timeConsuming;

	boolean hasArgs;

	public HeteroProfiler(String cmdStr) {
		super();

		kernelDatas = new ArrayList<KernelData>();
		timeConsuming = new TimeConsuming();
		hasArgs = false;

		if (cmdStr != null) {
			hasArgs = true;
			exeProfile(cmdStr);
		}
	}

	/**
	 * Launch the application.
	 * 
	 * @param args
	 */
	public static void main(String[] args) {
		String cmdStr = null;
		if (args.length > 0) {
			cmdStr = args[0];
			for (int i = 1; i < args.length; i++) {
				cmdStr += " " + args[i];
			}
		}
		try {
			HeteroProfiler window = new HeteroProfiler(cmdStr);
			window.open();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * Open the window.
	 */
	public void open() {
		Display display = Display.getDefault();
		createContents();
		
		int txtSize = 14;
		exeCmdText.setFont(new Font(display,"宋体",txtSize,SWT.NORMAL));
		profBtnButton.setFont(new Font(display,"宋体",txtSize,SWT.NORMAL));
		debugButton.setFont(new Font(display,"宋体",txtSize,SWT.NORMAL));
		tree.setFont(new Font(display,"宋体",txtSize,SWT.NORMAL));
		resultShowTab.setFont(new Font(display,"宋体",txtSize,SWT.NORMAL));
		txtLabel.setFont(new Font(display,"宋体",txtSize,SWT.NORMAL));

		center(display, shell);
		
		shell.pack();

		shell.open();
		shell.layout();

		if (hasArgs) {
			readProfResults(kernelDatas, timeConsuming);
			createTreeItems();
		}

		while (!shell.isDisposed()) {
			if (!display.readAndDispatch()) {
				display.sleep();
			}
		}
	}

	/**
	 * Create contents of the window.
	 */
	protected void createContents() {
		shell = new Shell();
		shell.setMinimumSize(new Point(800, 750));
		shell.setText("HetProfiler");
		shell.setLayout(new GridLayout(6, false));

		menuBar = new Menu(shell, SWT.BAR);	
		shell.setMenuBar(menuBar);

		createFileMenu();
		//createDebugMenu();
		createHelpMenu();
		createExeCmdTxt();
		createKernelTree();
		createResultShowTab();
		createTxtLabel();
	}

	/**
	 * Creates the File Menu.
	 * 
	 * @param parent
	 *            the parent menu
	 */
	private void createFileMenu() {
		Menu menu = new Menu(menuBar);
		MenuItem header = new MenuItem(menuBar, SWT.CASCADE);
		header.setText("&File");
		header.setMenu(menu);

		final MenuItem openMenu = new MenuItem(menu, SWT.PUSH);
		openMenu.setText("open");
		openMenu.addListener(SWT.Selection, new Listener() {
			@Override
			public void handleEvent(Event arg0) {
				FileDialog dialog = new FileDialog(shell);
				String data = dialog.open();
				if (data != null) {
					exeCmdText.setText(data);
				}
			}
		});
		
		MenuItem configMenu = new MenuItem(menu, SWT.PUSH);
		configMenu.setText("Configuration");
		configMenu.addListener(SWT.Selection, new Listener() {
			@Override
			public void handleEvent(Event arg0) {
				// TODO Auto-generated method stub
				MtrConfDialog dialog = new MtrConfDialog(shell);
				dialog.open();
			}
			
		});

		MenuItem closeMenu = new MenuItem(menu, SWT.PUSH);
		closeMenu.setText("close");
		closeMenu.addListener(SWT.Selection, new Listener() {
			@Override
			public void handleEvent(Event arg0) {
				shell.close();
			}
		});
	}
	
	/**
	 * Creates the debug Menu.
	 * 
	 * @param parent
	 *            the parent menu
	 */
	private void createDebugMenu() {
		Menu menu = new Menu(menuBar);
		MenuItem header = new MenuItem(menuBar, SWT.CASCADE);
		header.setText("&Debug");
		header.setMenu(menu);

		final MenuItem cpu = new MenuItem(menu, SWT.PUSH);
		cpu.setText("CPU Program");
		cpu.addListener(SWT.Selection, new Listener() {
			@Override
			public void handleEvent(Event arg0) {
				
				// TODO
				FileDialog dialog = new FileDialog(shell);
				String data = dialog.open();
				if (data != null) {
					exeCmdText.setText(data);
				}
			}
		});
		
		MenuItem gpu = new MenuItem(menu, SWT.PUSH);
		gpu.setText("GPU Program");
		gpu.addListener(SWT.Selection, new Listener() {
			@Override
			public void handleEvent(Event arg0) {
				// TODO 
				
				
				MtrConfDialog dialog = new MtrConfDialog(shell);
				dialog.open();
			}
			
		});
	}

	/**
	 * Creates the Help Menu.
	 * 
	 * @param parent
	 *            the parent menu
	 */
	private void createHelpMenu() {
		Menu menu = new Menu(menuBar);
		MenuItem header = new MenuItem(menuBar, SWT.CASCADE);
		header.setText("&Help");
		header.setMenu(menu);

		MenuItem item = new MenuItem(menu, SWT.PUSH);
		item.setText("about");
	}

	/**
	 * Creates the text widget to input executing command.
	 */
	private void createExeCmdTxt() {
		exeCmdText = new Text(shell, SWT.BORDER);
		profBtnButton = new Button(shell, SWT.PUSH);
		debugButton = new Button(shell, SWT.PUSH);
		GridData txtGridData = new GridData(SWT.FILL, SWT.TOP, true, false, 4,
				1);
		GridData btProfGridData = new GridData(SWT.FILL, SWT.TOP, false, false, 1,
				1);
		GridData btDebugGridData = new GridData(SWT.FILL, SWT.TOP, false, false, 1,
				1);
		btProfGridData.widthHint = 60;
		btDebugGridData.widthHint = 60;
		
		exeCmdText.setLayoutData(txtGridData);
		profBtnButton.setText("Profile");
		profBtnButton.setLayoutData(btProfGridData);

		profBtnButton.addSelectionListener(new SelectionAdapter() {
			@Override
			public void widgetSelected(SelectionEvent arg0) {
				String cmdLineStr = exeCmdText.getText();

				if (cmdLineStr != null && !"".equals(cmdLineStr)) {
					timeConsuming = new TimeConsuming(); // reset time
															// consuming.
					kernelDatas.clear(); // reset profile data.
					clearTreeItems();

					exeProfile(cmdLineStr);

					// read result and show.
					readProfResults(kernelDatas, timeConsuming);
					createTreeItems();
				}
			}
		});
		
		debugButton.setText("Debug");
		debugButton.setLayoutData(btDebugGridData);

		debugButton.addSelectionListener(new SelectionAdapter() {
			@Override
			public void widgetSelected(SelectionEvent arg0) {
				String cmdLineStr = exeCmdText.getText();

				//if (cmdLineStr != null && !"".equals(cmdLineStr)) {
					try{
						//Process process = Runtime.getRuntime().exec("xterm cuda-gdb");
						//String cmd = "xterm cuda-gdb /" + cmdLineStr;
						String cmd = "xterm cuda-gdb";
						System.out.println(cmd);
						Process process = Runtime.getRuntime().exec(cmd);
					} catch(Exception e) {
						e.printStackTrace();
					}
				//}
			}
		});

	}

	/**
	 * Creates the kernel tree widget.
	 */
	private void createKernelTree() {
		tree = new Tree(shell, SWT.BORDER);
		GridData treeGridData = new GridData(SWT.LEFT, SWT.FILL, false, true,
				1, 2);
		treeGridData.widthHint = 100;
		tree.setLayoutData(treeGridData);
	}

	/**
	 * Creates the label widget to show image of analysis result.
	 */
	private void createResultShowTab() {
		resultShowTab = new TabFolder(shell, SWT.BORDER);
		resultShowTab.setLayoutData(new GridData(SWT.FILL, SWT.FILL, true,
				true, 5, 1));

		chartTabItem = new TabItem(resultShowTab, SWT.NONE);
		chartTabItem.setText("Analysis");

		sheetTabItem = new TabItem(resultShowTab, SWT.NONE);
		sheetTabItem.setText("Result");
	}

	/**
	 * Creates the label widget to show analysis result in text.
	 */
	private void createTxtLabel() {
		txtLabel = new Text(shell, SWT.MULTI | SWT.BORDER | SWT.READ_ONLY | SWT.WRAP | SWT.V_SCROLL);
		GridData txtGridData = new GridData(SWT.FILL, SWT.BOTTOM, true, false,
				5, 1);
		txtGridData.heightHint = 100;
		txtLabel.setLayoutData(txtGridData);
		txtLabel.setText("");
	}

	/**
	 * Execute the profiling program
	 * 
	 * @param cmdLineStr
	 *            command line to execute the target program.
	 */
	private void exeProfile(String cmdLineStr) {
		try {
			String cmdStr = "./cudaProf " + cmdLineStr;
			System.out.println(cmdStr);
			Process process = Runtime.getRuntime().exec(cmdStr);
//					"./cudaProf " + cmdLineStr);
			process.waitFor();
			String proRes = readInputFromProcess(process);
			System.out.println(proRes);
		} catch (IOException e) {
			e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * Read input from Process
	 * 
	 * @param pro
	 *            the process to read
	 * @return the string output from the process
	 * @throws Exception
	 */
	private String readInputFromProcess(Process pro) throws Exception {
		StringBuffer sb = new StringBuffer();
		String line = null;
		BufferedReader reader = new BufferedReader(new InputStreamReader(
				pro.getInputStream()));
		try {
			while ((line = reader.readLine()) != null) {
				sb.append(line).append("\r\n");
			}
		} catch (IOException e) {
			e.printStackTrace();
			throw new RuntimeException("Fail to read the output of the process");
		} finally {
			reader.close();
		}
		return sb.toString();
	}

	/**
	 * read profile results and return kernels data.
	 * 
	 * @param kernelDatas
	 *            the metric values of each kernel, which will be set in this
	 *            function
	 */
	private void readProfResults(ArrayList<KernelData> kernelDatas,
			TimeConsuming timeCons) {
		File metricFile = new File("profile_kernel_data_result_summary.csv");
		File timeConsFile = new File("profile_trace_summary.csv");
		if (!metricFile.exists() || metricFile.isDirectory()
				|| !timeConsFile.exists() || timeConsFile.isDirectory()) {
			try {
				throw new FileNotFoundException();
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			}
		}
		try {
			// read time consuming
			BufferedReader timeConsReader = new BufferedReader(new FileReader(
					timeConsFile));
			String line = timeConsReader.readLine(); // read the header of the
														// csv file
			line = timeConsReader.readLine();
			while (line != null) {
				addTimeConsItem(line);
				line = timeConsReader.readLine();
			}
			timeConsReader.close();

			// read metrics
			BufferedReader metricReader = new BufferedReader(new FileReader(
					metricFile));
			line = metricReader.readLine(); // read the header of the csv file
			line = metricReader.readLine();
			while (line != null) {
				KernelData newKerData = createKernelData(line);
				kernelDatas.add(newKerData);

				line = metricReader.readLine();
			}
			metricReader.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private void addTimeConsItem(String line) {
		StringTokenizer tokens = new StringTokenizer(line, "\"");
		String typeStr = tokens.nextToken();
		String timeStr = tokens.nextToken();
		StringTokenizer timeTokens = new StringTokenizer(timeStr, ",");
		Long timeCons = Long.parseLong(timeTokens.nextToken()) * 1000000000
				+ Long.parseLong(timeTokens.nextToken());
		if (typeStr.equals("kernel")) {
			Long origValueObj = timeConsuming.getKerTimeCons().get("Kernel Execution");
			long origValueL = origValueObj == null ? 0 : origValueObj;
			timeConsuming.addKerTimeCons("Kernel Execution", origValueL + timeCons);
		} else if (typeStr.equals("memcpy")) {
			Long origValueObj = timeConsuming.getCpyTimeCons().get("Memory Copy");
			long origValueL = origValueObj == null ? 0 : origValueObj;
			
			timeConsuming.addCpyTimeCons("Memory Copy", origValueL + timeCons);
		} else if (typeStr.equals("cpu")) {
			timeConsuming.setCpuTimeCons(timeCons);
		}
	}

	private KernelData createKernelData(String line) {
		line = line.replaceAll("\"null\"", "0");
		System.out.println(line);

		String kerName;

		// basic info
		int device, gridX, gridY, gridZ, blockX, blockY, blockZ, dynamicSharedMem, staticSharedMem, localMemPerThread, localMemTotal, regPerThread, cacheConfigReq, cacheConfigUsed, sharedMemConfigUsed;
		
		int theoryLimiter, achievedLimiter;

		// utilization
		float theory_occupancy, achieved_occupancy, dram_utilization, l1_shared_utilization, l2_utilization, ldst_fu_utilization, alu_fu_utilization;

		// efficiency
		float sm_efficiency, gld_efficiency, gst_efficiency, warp_execution_efficiency, warp_nonpred_execution_efficiency, shared_efficiency;

		// overhead
		float global_replay_overhead, local_memory_overhead, shared_replay_overhead;

		// throughput
		long gld_throughput, gst_throughput, dram_read_throughput, dram_write_throughput, ldst_executed, inst_fp_32, inst_fp_64, inst_integer, shared_load_replay, shared_store_replay, l2_total_misses, l2_total_hit, l1_global_load_miss, l1_global_load_hit, inst_executed, shared_load, shared_store, local_load, local_store, gld_request, gst_request;

		// others
		float ipc, l1_cache_global_hit_rate;

		StringTokenizer tokens = new StringTokenizer(line, "\"");

		kerName = tokens.nextToken();
		KernelData newKerData = new KernelData();
		newKerData.setName(kerName);

		String metricStr = tokens.nextToken().substring(1);
		tokens = new StringTokenizer(metricStr, ",");
		

		device = Integer.parseInt(tokens.nextToken());
		gridX = Integer.parseInt(tokens.nextToken());
		gridY = Integer.parseInt(tokens.nextToken());
		gridZ = Integer.parseInt(tokens.nextToken());
		blockX = Integer.parseInt(tokens.nextToken());
		blockY = Integer.parseInt(tokens.nextToken());
		blockZ = Integer.parseInt(tokens.nextToken());
		dynamicSharedMem = Integer.parseInt(tokens.nextToken());
		staticSharedMem = Integer.parseInt(tokens.nextToken());
		localMemPerThread = Integer.parseInt(tokens.nextToken());
		localMemTotal = Integer.parseInt(tokens.nextToken());
		regPerThread = Integer.parseInt(tokens.nextToken());
		cacheConfigReq = Integer.parseInt(tokens.nextToken());
		cacheConfigUsed = Integer.parseInt(tokens.nextToken());
		sharedMemConfigUsed = Integer.parseInt(tokens.nextToken());
		theoryLimiter = Integer.parseInt(tokens.nextToken()); // theory_limiter
		achievedLimiter = Integer.parseInt(tokens.nextToken()); // achieved_limiter
		theory_occupancy = Float.parseFloat(tokens.nextToken());
		achieved_occupancy = Float.parseFloat(tokens.nextToken());
		sm_efficiency = Float.parseFloat(tokens.nextToken());
		gld_efficiency = Float.parseFloat(tokens.nextToken());
		gst_efficiency = Float.parseFloat(tokens.nextToken());

		global_replay_overhead = Float.parseFloat(tokens.nextToken());

		String tkStr = tokens.nextToken();
		System.out.println(tkStr);
		

		System.out.println(metricStr);

		gld_throughput = Long.parseLong(tkStr);
		gst_throughput = Long.parseLong(tokens.nextToken());
		local_memory_overhead = Float.parseFloat(tokens.nextToken());
		warp_execution_efficiency = Float.parseFloat(tokens.nextToken());
		warp_nonpred_execution_efficiency = Float
				.parseFloat(tokens.nextToken());
		shared_efficiency = Float.parseFloat(tokens.nextToken());
		shared_replay_overhead = Float.parseFloat(tokens.nextToken());
		dram_read_throughput = Long.parseLong(tokens.nextToken());
		dram_write_throughput = Long.parseLong(tokens.nextToken());
		dram_utilization = Float.parseFloat(tokens.nextToken());
		ldst_executed = Long.parseLong(tokens.nextToken());
		inst_fp_32 = Long.parseLong(tokens.nextToken());
		inst_fp_64 = Long.parseLong(tokens.nextToken());
		inst_integer = Long.parseLong(tokens.nextToken());
		ipc = Float.parseFloat(tokens.nextToken());
		l1_cache_global_hit_rate = Float.parseFloat(tokens.nextToken());
		l1_shared_utilization = Float.parseFloat(tokens.nextToken());
		l2_utilization = Float.parseFloat(tokens.nextToken());
		ldst_fu_utilization = Float.parseFloat(tokens.nextToken());
		alu_fu_utilization = Float.parseFloat(tokens.nextToken());
		shared_load_replay = Long.parseLong(tokens.nextToken());
		shared_store_replay = Long.parseLong(tokens.nextToken());
		l2_total_misses = Long.parseLong(tokens.nextToken());
		l2_total_hit = Long.parseLong(tokens.nextToken());
		l1_global_load_miss = Long.parseLong(tokens.nextToken());
		l1_global_load_hit = Long.parseLong(tokens.nextToken());
		inst_executed = Long.parseLong(tokens.nextToken());
		shared_load = Long.parseLong(tokens.nextToken());
		shared_store = Long.parseLong(tokens.nextToken());
		local_load = Long.parseLong(tokens.nextToken());
		local_store = Long.parseLong(tokens.nextToken());
		gld_request = Long.parseLong(tokens.nextToken());
		gst_request = Long.parseLong(tokens.nextToken());

		newKerData.addBasicInfoItem("device", device);
		newKerData.addBasicInfoItem("gridX", gridX);
		newKerData.addBasicInfoItem("gridY", gridY);
		newKerData.addBasicInfoItem("gridZ", gridZ);
		newKerData.addBasicInfoItem("blockX", blockX);
		newKerData.addBasicInfoItem("blockY", blockY);
		newKerData.addBasicInfoItem("blockZ", blockZ);
		newKerData.addBasicInfoItem("dynamicSharedMem", dynamicSharedMem);
		newKerData.addBasicInfoItem("staticSharedMem", staticSharedMem);
		newKerData.addBasicInfoItem("localMemPerThread", localMemPerThread);
		newKerData.addBasicInfoItem("localMemTotal", localMemTotal);
		newKerData.addBasicInfoItem("regPerThread", regPerThread);
		newKerData.addBasicInfoItem("cacheConfigReq", cacheConfigReq);
		newKerData.addBasicInfoItem("cacheConfigUsed", cacheConfigUsed);
		newKerData.addBasicInfoItem("sharedMemConfigUsed", sharedMemConfigUsed);

		newKerData.addOccupancyItem("theory_occupancy", theory_occupancy);
		newKerData.addOccupancyItem("achieved_occupancy", achieved_occupancy);

		newKerData.addEfficiencyItem("sm_efficiency", sm_efficiency);
		newKerData.addEfficiencyItem("gld_efficiency", gld_efficiency);
		newKerData.addEfficiencyItem("gst_efficiency", gst_efficiency);
		newKerData.addEfficiencyItem("warp_execution_efficiency",
				warp_execution_efficiency);
		newKerData.addEfficiencyItem("warp_nonpred_execution_efficiency",
				warp_nonpred_execution_efficiency);
		newKerData.addEfficiencyItem("shared_efficiency", shared_efficiency);

		newKerData.addUtilizationItem("dram_utilization", dram_utilization);
		newKerData.addUtilizationItem("l1_shared_utilization",
				l1_shared_utilization);
		newKerData.addUtilizationItem("l2_utilization", l2_utilization);
		newKerData.addUtilizationItem("ldst_fu_utilization",
				ldst_fu_utilization);
		newKerData.addUtilizationItem("alu_fu_utilization", alu_fu_utilization);

		newKerData.addOverheadItem("global_replay_overhead",
				global_replay_overhead);
		newKerData.addOverheadItem("local_memory_overhead",
				local_memory_overhead);
		newKerData.addOverheadItem("shared_replay_overhead",
				shared_replay_overhead);

		newKerData.addInstructionItem("gld_throughput", gld_throughput);
		newKerData.addInstructionItem("gst_throughput", gst_throughput);
		newKerData.addInstructionItem("dram_read_throughput",
				dram_read_throughput);
		newKerData.addInstructionItem("dram_write_throughput",
				dram_write_throughput);
		newKerData.addInstructionItem("ldst_executed", ldst_executed);
		newKerData.addInstructionItem("inst_fp_32", inst_fp_32);
		newKerData.addInstructionItem("inst_fp_64", inst_fp_64);
		newKerData.addInstructionItem("inst_integer", inst_integer);
		newKerData.addInstructionItem("shared_load_replay", shared_load_replay);
		newKerData.addInstructionItem("shared_store_replay",
				shared_store_replay);
		newKerData.addInstructionItem("l2_total_misses", l2_total_misses);
		newKerData.addInstructionItem("l2_total_hit", l2_total_hit);
		newKerData.addInstructionItem("l1_global_load_miss",
				l1_global_load_miss);
		newKerData.addInstructionItem("l1_global_load_hit", l1_global_load_hit);
		newKerData.addInstructionItem("inst_executed", inst_executed);
		newKerData.addInstructionItem("shared_load", shared_load);
		newKerData.addInstructionItem("shared_store", shared_store);
		newKerData.addInstructionItem("local_load", local_load);
		newKerData.addInstructionItem("local_store", local_store);
		newKerData.addInstructionItem("gld_request", gld_request);
		newKerData.addInstructionItem("gst_request", gst_request);

		newKerData.addOthersItem("ipc", ipc);
		newKerData.addOthersItem("l1_cache_global_hit_rate",
				l1_cache_global_hit_rate);
		
		newKerData.setTheoryLimiter(theoryLimiter);
		newKerData.setAchievedLimiter(achievedLimiter);

		return newKerData;
	}

	/**
	 * Creates the tree to show kernels and metric items
	 */
	protected void createTreeItems() {
		final TreeItem timeConsumeItem = new TreeItem(tree, 0);
		timeConsumeItem.setText("time consuming");

		final TreeItem kernelItem = new TreeItem(tree, 0);
		kernelItem.setText("kernels");

		final String childItemsString[] = { "basicInfo", "occupancy",
				"efficiency", "utilization", "instruction" };
		// final String childItemsString[] = { "basicInfo", "occupancy",
		// 		"efficiency", "utilization", "overhead", "instruction" };

		for (KernelData kerData : kernelDatas) {
			TreeItem item = new TreeItem(kernelItem, 0);
			item.setText(kerData.getName());
			for (int i = 0; i < childItemsString.length; i++) {
				final String mtrClass = childItemsString[i];
				TreeItem subItem = new TreeItem(item, 0);
				subItem.setText(mtrClass);
			}
		}

		tree.addListener(SWT.DefaultSelection, new Listener() {

			public void handleEvent(Event event) {
				if (event.item instanceof TreeItem) {
					TreeItem itemSelected = (TreeItem) event.item;
					if (itemSelected == timeConsumeItem) {
						timeConsumeShow();
					} else if (itemSelected.getParentItem() != null
							&& itemSelected.getParentItem().getParentItem() == kernelItem) {
						String kerName = itemSelected.getParentItem().getText();
						String mtrClass = itemSelected.getText();
						metricSelected(kerName, mtrClass);
					} else {
						boolean expanded = itemSelected.getExpanded();
						itemSelected.setExpanded(!expanded);
					}
				}
			}
		});

		tree.setSelection(timeConsumeItem);
	}

	private void timeConsumeShow() {

		JFreeChart pie = ChartFactory.createPieChart("Time Consuming",
				createPieDataset(timeConsuming));

		Composite composite = new Composite(resultShowTab, SWT.NO_BACKGROUND
				| SWT.EMBEDDED);
		Frame pieFrame = SWT_AWT.new_Frame(composite);
		pieFrame.add(new ChartPanel(pie));

		if (chartTabItem.isDisposed()) {
			chartTabItem = new TabItem(resultShowTab, SWT.NONE);
			chartTabItem.setText("Analysis");
		}
		chartTabItem.setControl(composite);

		//
		// Add sheet.
		Table table = new Table(resultShowTab, SWT.MULTI | SWT.FULL_SELECTION
				| SWT.CHECK);
		GridData tableGrid = new GridData();
		tableGrid.horizontalAlignment = SWT.FILL;
		tableGrid.grabExcessHorizontalSpace = true;
		tableGrid.grabExcessVerticalSpace = true;
		tableGrid.verticalAlignment = SWT.FILL;
		table.setHeaderVisible(true);
		// table.setLayoutData(tableGrid);
		table.setLinesVisible(true);
		sheetTabItem.setControl(table);

		String[] tableHeader = new String[1
				+ timeConsuming.getKerTimeCons().size()
				+ timeConsuming.getCpyTimeCons().size()];
		String[] values = new String[tableHeader.length];
		tableHeader[0] = "CPU Time";
		values[0] = String.valueOf(timeConsuming.getCpuTimeCons());
		int i = 1;
		Enumeration<String> enuKer = timeConsuming.getKerTimeCons().keys();
		while (enuKer.hasMoreElements()) {
			String key = enuKer.nextElement();
			tableHeader[i] = key;
			values[i] = timeConsuming.getKerTimeCons().get(key).toString();
			i++;
		}
		Enumeration<String> enuCpy = timeConsuming.getCpyTimeCons().keys();
		while (enuCpy.hasMoreElements()) {
			String key = enuCpy.nextElement();
			tableHeader[i] = key;
			values[i] = timeConsuming.getCpyTimeCons().get(key).toString();
			i++;
		}

		for (String headerItem : tableHeader) {
			TableColumn tableColumn = new TableColumn(table, SWT.CENTER);
			tableColumn.setText(headerItem);
			tableColumn.pack();
		}
		TableItem item = new TableItem(table, SWT.NONE);
		item.setText(values);

		pieFrame.setVisible(true);
	}

	private DefaultPieDataset createPieDataset(TimeConsuming timeConsuming) {
		DefaultPieDataset dftDataset = new DefaultPieDataset();

		Hashtable<String, Long> kernelTimes = timeConsuming.getKerTimeCons();
		Hashtable<String, Long> cpyTimes = timeConsuming.getCpyTimeCons();

		Enumeration<String> timeEnum;

		timeEnum = kernelTimes.keys();
		while (timeEnum.hasMoreElements()) {
			String key = timeEnum.nextElement();
			dftDataset.setValue(key, kernelTimes.get(key));
		}
		timeEnum = cpyTimes.keys();
		while (timeEnum.hasMoreElements()) {
			String key = timeEnum.nextElement();
			dftDataset.setValue(key, cpyTimes.get(key));
		}
		dftDataset.setValue("CPU Execution", timeConsuming.getCpuTimeCons());

		return dftDataset;
	}

	/**
	 * Executes when a metric class in the kernel tree has been selected.
	 * 
	 * @param kerName
	 *            kernel's name
	 * @param mtrClass
	 *            the metric class
	 */
	protected void metricSelected(String kerName, String mtrClass) {

		if (!(mtrClass.equals("basicInfo") || mtrClass.equals("occupancy")
				|| mtrClass.equals("efficiency")
				|| mtrClass.equals("utilization")
				|| mtrClass.equals("overhead") || mtrClass
					.equals("instruction"))) {
			return;
		}
		txtLabel.setText("");
		
		String occupancyLimiter[] = {"None", "Block Size", "Regeters Per Thread", "SMem Per Block", "Grid Size"};

		Hashtable<String, Integer> mtrDataInteger = null;
		Hashtable<String, Float> mtrDataFloat = null;
		Hashtable<String, Long> mtrDataLong = null;
		boolean showChart = true;
		for (KernelData kerData : kernelDatas) {
			if (kerData.getName().equals(kerName)) {
				if (mtrClass.equals("basicInfo")) {
					mtrDataInteger = kerData.getBasicInfo();
					showChart = false;
				} else if (mtrClass.equals("occupancy")) {
					mtrDataFloat = kerData.getOccupancy();
				} else if (mtrClass.equals("efficiency")) {
					mtrDataFloat = kerData.getEfficiency();
				} else if (mtrClass.equals("utilization")) {
					mtrDataFloat = kerData.getUtilization();
				} else if (mtrClass.equals("overhead")) {
					mtrDataFloat = kerData.getOverhead();
				} else if (mtrClass.equals("instruction")) {
					mtrDataLong = kerData.getInstruction();
					showChart = false;
				}

				// Add chart.
				if (showChart == true) {
					if (chartTabItem.isDisposed()) {
						chartTabItem = new TabItem(resultShowTab, SWT.NONE);
						chartTabItem.setText("Analysis");
					}
					resultShowTab.setSelection(chartTabItem);

					JFreeChart chart = null;

					if (mtrClass.equals("basicInfo")) {
						chart = ChartFactory.createBarChart(mtrClass,
								"Metric Names", "Values",
								createChartDataset(mtrDataInteger));
					} else if (mtrClass.equals("occupancy")
							|| mtrClass.equals("efficiency")
							|| mtrClass.equals("utilization")
							|| mtrClass.equals("overhead")) {
						chart = ChartFactory.createBarChart(mtrClass,
								"Metric Names", "Values",
								createChartDataset(mtrDataFloat));
						
						if(mtrClass.equals("occupancy")) {
							String occuStr = "";
							float theory = mtrDataFloat.get("theory_occupancy");
							float achieved = mtrDataFloat.get("achieved_occupancy");
							if(theory < 0.8) {
								occuStr += "Theory Occupancy is too low, which is " + theory + ". ";
								int theoryLimiter = kerData.getTheoryLimiter();
								if(theoryLimiter > 0) {
									occuStr += " The limiter is " + occupancyLimiter[theoryLimiter] + ".\n";
								}
							}
							if(achieved + 0.05 < theory && achieved < 0.8) {
								occuStr += "Achieved Occupancy is too low, which is " + achieved + ". ";
								int achievedLimiter = kerData.getAchievedLimiter();
								if(achievedLimiter > 0) {
									occuStr += " The limiter is " + occupancyLimiter[achievedLimiter] + ".\n";
								}
							}
							if(!occuStr.equals("")) {
								txtLabel.setText(occuStr);
								txtLabel.setForeground(new Color(null, 255, 0, 0));
							}
						}
						
						if(mtrClass.equals("efficiency")) {
							String effiStr = "";
							double smEffi = mtrDataFloat.get("sm_efficiency");
							double gldEffi = mtrDataFloat.get("gld_efficiency");
							double gstEffi = mtrDataFloat.get("gst_efficiency");
							double weEffi = mtrDataFloat.get("warp_execution_efficiency");
							double wneEffi = mtrDataFloat.get("warp_nonpred_execution_efficiency");
							double smemEffi = mtrDataFloat.get("shared_efficiency");
							if(smEffi > 0.00001 && smEffi < 50) {
								effiStr += "sm_efficiency is too low. Try to use parallel resources better. Try to increase occupancy.\n";
							}
							if(smemEffi > 0.00001 && smemEffi < 50) {
								effiStr += "shared_efficiency is too low. Bank conflict exists.\n";
							}
							if(gldEffi > 0.00001 && gldEffi < 50) {
								effiStr += "gld_efficiency is too low. Try to optimize memory access mode.\n";
							}
							if(gstEffi > 0.00001 && gstEffi < 50) {
								effiStr += "gst_efficiency is too low. Try to optimize memory access mode.\n";
							}
							if(weEffi > 0.00001 && weEffi < 50) {
								effiStr += "warp_execution_efficiency is too low. Warp divergence exists.\n";
							}
							if(wneEffi > 0.00001 && wneEffi < 50) {
								effiStr += "warp_nonpred_execution_efficiency is too low. Warp divergence exists.\n";
							}
							if(!effiStr.equals("")) {
								txtLabel.setText(effiStr);
								txtLabel.setForeground(new Color(null, 255, 0, 0));
							}
						}
						
					} else if (mtrClass.equals("instruction")) {
						chart = ChartFactory.createBarChart(mtrClass,
								"Metric Names", "Values",
								createChartDataset(mtrDataLong));
					}

					Composite composite = new Composite(resultShowTab,
							SWT.NO_BACKGROUND | SWT.EMBEDDED);
					Frame chartFrame = SWT_AWT.new_Frame(composite);
					chartFrame.add(new ChartPanel(chart));
					chartTabItem.setControl(composite);

					chartFrame.setVisible(true);
				} else {
					if (!chartTabItem.isDisposed()) {
						chartTabItem.dispose();
					}
				}

				// Add sheet.
				Table table = new Table(resultShowTab, SWT.MULTI
						| SWT.FULL_SELECTION | SWT.CHECK);
				GridData tableGrid = new GridData();
				tableGrid.horizontalAlignment = SWT.FILL;
				tableGrid.grabExcessHorizontalSpace = true;
				tableGrid.grabExcessVerticalSpace = true;
				tableGrid.verticalAlignment = SWT.FILL;
				table.setHeaderVisible(true);
				// table.setLayoutData(tableGrid);
				table.setLinesVisible(true);
				sheetTabItem.setControl(table);

				String[] tableHeader = null;
				if (mtrClass.equals("basicInfo")) {
					tableHeader = Collections.list(mtrDataInteger.keys())
							.toArray(new String[0]);
				} else if (mtrClass.equals("occupancy")
						|| mtrClass.equals("efficiency")
						|| mtrClass.equals("utilization")
						|| mtrClass.equals("overhead")) {
					tableHeader = Collections.list(mtrDataFloat.keys())
							.toArray(new String[0]);
				} else if (mtrClass.equals("instruction")) {
					tableHeader = Collections.list(mtrDataLong.keys()).toArray(
							new String[0]);
				}

				String[] values = new String[tableHeader.length];
				int i = 0;
				for (String headerItem : tableHeader) {

					TableColumn tableColumn = new TableColumn(table, SWT.CENTER);
					tableColumn.setText(headerItem);
					tableColumn.pack();

					if (mtrClass.equals("basicInfo")) {
						values[i++] = mtrDataInteger.get(headerItem).toString();
					} else if (mtrClass.equals("occupancy")
							|| mtrClass.equals("efficiency")
							|| mtrClass.equals("utilization")
							|| mtrClass.equals("overhead")) {
						values[i++] = mtrDataFloat.get(headerItem).toString();
					} else if (mtrClass.equals("instruction")) {
						values[i++] = mtrDataLong.get(headerItem).toString();
					}
				}

				TableItem item = new TableItem(table, SWT.NONE);
				item.setText(values);

				break;
			}
		}
	}

	/**
	 * Clears the kernel tree.
	 */
	protected void clearTreeItems() {
		tree.removeAll();
	}

	private <T extends Number> DefaultCategoryDataset createChartDataset(
			Hashtable<String, T> mtrData) {
		DefaultCategoryDataset dftDataset = new DefaultCategoryDataset();

		Enumeration<String> mtrEnum = mtrData.keys();
		while (mtrEnum.hasMoreElements()) {
			String key = mtrEnum.nextElement();
			T value = mtrData.get(key);
			if (value instanceof Integer) {
				dftDataset.addValue((Integer) value, key, "metrics");
			} else if (value instanceof Float) {
				dftDataset.addValue((Float) value, key, "metrics");
			} else if (value instanceof Long) {
				dftDataset.addValue((Long) value, key, "metrics");
			}
		}

		return dftDataset;
	}
	
	/**
	 * 设置窗口位于屏幕中间
	 * 
	 * @param display
	 *            设备
	 * @param shell
	 *            要调整位置的窗口对象
	 */
	public static void center(Display display, Shell shell) {
		Rectangle bounds = display.getPrimaryMonitor().getBounds();
		Rectangle rect = shell.getBounds();
		int x = bounds.x + (bounds.width - rect.width) / 2;
		int y = bounds.y + (bounds.height - rect.height) / 2;
		shell.setLocation(x, y);
	}

}
