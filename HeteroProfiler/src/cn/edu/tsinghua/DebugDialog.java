/**
 * 
 */
package cn.edu.tsinghua;

import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.Font;
import org.eclipse.swt.graphics.Rectangle;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.Button;
import org.eclipse.swt.widgets.Dialog;
import org.eclipse.swt.widgets.Display;
import org.eclipse.swt.widgets.Event;
import org.eclipse.swt.widgets.Listener;
import org.eclipse.swt.widgets.Shell;
import org.eclipse.swt.widgets.Table;
import org.eclipse.swt.widgets.TableColumn;
import org.eclipse.swt.widgets.TableItem;

/**
 * @author zhengzhen
 *
 */
public class DebugDialog extends Dialog {

	public DebugDialog(Shell parent) {
		super(parent);
		// TODO Auto-generated constructor stub
	}

	public String open() {
		Shell parent = this.getParent();
		final Shell dialog = new Shell(parent, SWT.DIALOG_TRIM
				| SWT.APPLICATION_MODAL);
		dialog.setText("Debugging");
		dialog.setSize(300, 300);
		GridLayout dialogLayout = new GridLayout();
		dialogLayout.numColumns = 1;
		dialog.setLayout(dialogLayout);
		Display display = parent.getDisplay();
		center(display, dialog);

		dialog.open();

		int txtSize = 14;
		Table table = new Table(dialog, SWT.MULTI | SWT.FULL_SELECTION
				| SWT.CHECK);
		/*
		GridData tableGridData = new GridData(SWT.FILL, SWT.TOP, true, false,
				2, 1);
		table.setLayoutData(tableGridData);
		*/
		table.setFont(new Font(display,"宋体",txtSize,SWT.NORMAL));
		table.setHeaderVisible(true);
		
		TableColumn tbCol = new TableColumn(table, SWT.FILL);
		tbCol.setText("Metrics Classification");
		tbCol.setWidth(290);
		// tbCol.pack();
		TableItem item1 = new TableItem(table, SWT.NONE);
		TableItem item2 = new TableItem(table, SWT.NONE);
		TableItem item3 = new TableItem(table, SWT.NONE);
		TableItem item4 = new TableItem(table, SWT.NONE);
		TableItem item5 = new TableItem(table, SWT.NONE);
		TableItem item6 = new TableItem(table, SWT.NONE);
		item1.setText("Basic Information");
		item2.setText("Occupancy");
		item3.setText("Shared Access Efficiency");
		item4.setText("Global Access Efficiency");
		item5.setText("Utilization");
		item6.setText("Instructions");
		item1.setChecked(true);
		item2.setChecked(true);
		item3.setChecked(true);
		item4.setChecked(true);
		item5.setChecked(true);
		item6.setChecked(true);
		table.pack();

		Button okBtn = new Button(dialog, SWT.PUSH);
		/*
		GridData btnGridData = new GridData(SWT.FILL, SWT.BOTTOM, false, false,
				1, 1);
		okBtn.setLayoutData(btnGridData);
		*/
		okBtn.setText("Finish");
		okBtn.setSize(80, 40);
		okBtn.setLocation(110, 220);
		
		okBtn.setFont(new Font(display,"宋体",txtSize,SWT.NORMAL));
		
		okBtn.addListener(SWT.Selection, new Listener() {

			@Override
			public void handleEvent(Event arg0) {
				// TODO Auto-generated method stub
				dialog.dispose();
			}
			
		});

		while (!dialog.isDisposed()) {
			if (display.readAndDispatch()) {
				display.sleep();
			}
		}
		return "closed";
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
