package com.support.automate.jarvis;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.Statement;
import java.util.Properties;

public class ConnectPostgres {

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		String url = "jdbc:postgresql://localhost:5432/postgres";
		Properties props = new Properties();
		props.setProperty("user", "postgres");
		props.setProperty("password", "admin");
		props.setProperty("ssl", "false");
		Connection conn = DriverManager.getConnection(url, props);

		System.out.println("Connection success");

		Statement stmt = null;
		stmt = conn.createStatement();
		
		String sql = "SELECT * FROM \"NLP_DATA\"";
		
		ResultSet rs = stmt.executeQuery(sql);
		// STEP 5: Extract data from result set
		while (rs.next()) {
			// Retrieve by column name
			String tags = rs.getString("tags");

			// Display values
			System.out.print("tags: " + tags);
		}
		rs.close();
	}

}
