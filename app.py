import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
import time
from datetime import date
import pickle
import itertools
import plotly.express as px
from plot_setup import finastra_theme
from download_data import Data
import sys
import metadata_parser
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq client with proper error handling
def initialize_groq_client():
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        st.warning("Groq API key not found. Some insights features will be disabled.")
        return None
    try:
        return Groq(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize Groq client: {str(e)}")
        return None

groq_client = initialize_groq_client()

def get_graph_insight(chart_data, chart_type):
    """Generate insights for a specific chart using Groq's LLaMA model."""
    if not groq_client:
        return "Insights not available - Groq API key not configured."
        
    prompt = f"""
    Analyze the following data for a {chart_type} and provide a brief, insightful observation:
    {chart_data}
    
    Instructions:
    - Provide 1-2 sentences of key insights
    - Focus on trends, patterns, or notable observations
    - Be specific and data-driven
    - Keep it concise and business-relevant
    """
    
    try:
        response = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert financial and ESG analyst skilled at interpreting data visualizations."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Unable to generate insight: {str(e)}"


# Set page config at the very beginning of the script
st.set_page_config(
    page_title="ESG AI",
    page_icon=os.path.join(".", "raw", "esg_ai_logo.png"),
    layout='centered',
    initial_sidebar_state="collapsed"
)


####### CACHED FUNCTIONS ######
@st.cache_data(show_spinner=False)
def filter_company_data(df_company, esg_categories, start, end):
    # Convert start and end to Timestamp
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    
    # Filter E,S,G Categories
    comps = []
    for i in esg_categories:
        X = df_company[df_company[i] == True]
        comps.append(X)
    df_company = pd.concat(comps)
    df_company = df_company[df_company.DATE.between(start, end)]
    return df_company

@st.cache_data(show_spinner=False)
def load_data(start_data, end_data):
    data = Data().read(start_data, end_data)
    # Ensure DATE column is datetime
    if 'data' in data and 'DATE' in data['data'].columns:
        data['data']['DATE'] = pd.to_datetime(data['data']['DATE'])
    companies = data["data"].Organization.sort_values().unique().tolist()
    companies.insert(0,"Select a Company")
    return data, companies

@st.cache_data(show_spinner=False)
def filter_publisher(df_company, publisher):
    if publisher != 'all':
        df_company = df_company[df_company['SourceCommonName'] == publisher]
    return df_company

def get_melted_frame(data_dict, frame_names, keepcol=None, dropcol=None):
    try:
        if keepcol:
            # Check if the keepcol exists in the DataFrame
            reduced = {k: df[keepcol].rename(k) for k, df in data_dict.items() 
                       if k in frame_names and keepcol in df.columns}
        else:
            # Check if the dropcol exists in the DataFrame
            reduced = {k: df.drop(columns=dropcol).mean(axis=1).rename(k) 
                       for k, df in data_dict.items() if k in frame_names and dropcol in df.columns}
        
        if not reduced:
            raise ValueError(f"Column '{keepcol if keepcol else dropcol}' not found in the dataset.")
        
        df = (pd.concat(list(reduced.values()), axis=1)
              .reset_index()
              .melt("date")
              .sort_values("date")
              .ffill())
        df.columns = ["DATE", "ESG", "Score"]
        return df.reset_index(drop=True)
    
    except Exception as e:
        st.error(f"Error in get_melted_frame: {str(e)}")
        return pd.DataFrame()  # Return an empty DataFrame to avoid further issues

def filter_on_date(df, start, end, date_col="DATE"):
    # Convert start and end to Timestamp
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    
    df = df[(df[date_col] >= start) & (df[date_col] <= end)]
    return df

@st.cache_data(show_spinner=False)
def get_clickable_name(url):
    try:
        T = metadata_parser.MetadataParser(url=url, search_head_only=True)
        title = T.metadata["og"]["title"].replace("|", " - ")
        return f"[{title}]({url})"
    except:
        return f"[{url}]({url})"

def main(start_data, end_data):
    ###### CUSTOMIZE COLOR THEME ######
    alt.themes.register("finastra", finastra_theme)
    alt.themes.enable("finastra")
    violet, fuchsia = ["#694ED6", "#C137A2"]

    ###### SET UP PAGE ######
    icon_path = os.path.join(".", "raw", "esg_ai_logo.png")
    _, logo, _ = st.columns(3)
    logo.image(icon_path, width=200)
    style = ("text-align:center; padding: 0px; font-family: arial black;, "
             "font-size: 400%")
    title = f"<h1 style='{style}'>ESG<sup>AI</sup></h1><br><br>"
    st.write(title, unsafe_allow_html=True)

    ###### LOAD DATA ######
    with st.spinner(text="Fetching Data..."):
        try:
            data, companies = load_data(start_data, end_data)
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return

    df_conn = data["conn"]
    df_data = data["data"]
    embeddings = data["embed"]

    ####### CREATE SIDEBAR CATEGORY FILTER######
    st.sidebar.title("Filter Options")
    date_place = st.sidebar.empty()
    esg_categories = st.sidebar.multiselect("Select News Categories",
                                          ["E", "S", "G"], ["E", "S", "G"])
    pub = st.sidebar.empty()
    num_neighbors = st.sidebar.slider("Number of Connections", 1, 20, value=8)

    ###### RUN COMPUTATIONS WHEN A COMPANY IS SELECTED ######
    company = st.selectbox("Select a Company to Analyze", companies)
    if company and company != "Select a Company":
        ###### FILTER ######
        df_company = df_data[df_data.Organization == company]
        diff_col = f"{company.replace(' ', '_')}_diff"
        esg_keys = ["E_score", "S_score", "G_score"]
        esg_df = get_melted_frame(data, esg_keys, keepcol=diff_col)
        ind_esg_df = get_melted_frame(data, esg_keys, dropcol="industry_tone")
        tone_df = get_melted_frame(data, ["overall_score"], keepcol=diff_col)
        ind_tone_df = get_melted_frame(data, ["overall_score"],
                                     dropcol="industry_tone")

        ####### DATE WIDGET ######
        if isinstance(df_company.DATE.min(), pd.Timestamp):
            min_date = df_company.DATE.min().date()
            max_date = df_company.DATE.max().date()
        else:
            min_date = df_company.DATE.min()
            max_date = df_company.DATE.max()

        selected_dates = date_place.date_input("Select a Date Range",
            value=[min_date, max_date], 
            min_value=min_date, 
            max_value=max_date, 
            key=None)
        time.sleep(0.8)
        start, end = selected_dates

        ###### FILTER DATA ######
        df_company = filter_company_data(df_company, esg_categories, start, end)
        esg_df = filter_on_date(esg_df, start, end)
        ind_esg_df = filter_on_date(ind_esg_df, start, end)
        tone_df = filter_on_date(tone_df, start, end)
        ind_tone_df = filter_on_date(ind_tone_df, start, end)
        date_filtered = filter_on_date(df_data, start, end)

        ###### PUBLISHER SELECT BOX ######
        publishers = df_company.SourceCommonName.sort_values().unique().tolist()
        publishers.insert(0, "all")
        publisher = pub.selectbox("Select Publisher", publishers)
        df_company = filter_publisher(df_company, publisher)

        ###### DISPLAY DATA ######
        with st.expander(f"View {company.title()} Data:", True):
            st.write(f"### {len(df_company):,d} Matching Articles for " +
                    company.title())
            display_cols = ["DATE", "SourceCommonName", "Tone", "Polarity",
                          "NegativeTone", "PositiveTone"]
            st.write(df_company[display_cols])

            st.write("#### Sample Articles")
            link_df = df_company[["DATE", "DocumentIdentifier"]].head(3).copy()
            link_df["ARTICLE"] = link_df.DocumentIdentifier.apply(get_clickable_name)
            link_df = link_df[["DATE", "ARTICLE"]].to_markdown(index=False)
            st.markdown(link_df)

        ###### CHART: METRIC OVER TIME ######
        st.markdown("---")
        col1, col2 = st.columns((1, 3))

        metric_options = ["Tone", "NegativeTone", "PositiveTone", "Polarity",
                         "ActivityDensity", "WordCount", "Overall Score",
                         "ESG Scores"]
        line_metric = col1.radio("Choose Metric", options=metric_options)

        if line_metric == "ESG Scores":
            # Get ESG scores
            esg_df["WHO"] = company.title()
            ind_esg_df["WHO"] = "Industry Average"
            esg_plot_df = pd.concat([esg_df, ind_esg_df]
                                  ).reset_index(drop=True)
            esg_plot_df.replace({"E_score": "Environment", "S_score": "Social",
                               "G_score": "Governance"}, inplace=True)

            metric_chart = alt.Chart(esg_plot_df, title="Trends Over Time"
                                  ).mark_line().encode(
                x=alt.X("yearmonthdate(DATE):O", title="DATE"),
                y=alt.Y("Score:Q"),
                color=alt.Color("ESG", sort=None, legend=alt.Legend(
                    title=None, orient="top")),
                strokeDash=alt.StrokeDash("WHO", sort=None, legend=alt.Legend(
                    title=None, symbolType="stroke", symbolFillColor="gray",
                    symbolStrokeWidth=4, orient="top")),
                tooltip=["DATE", "ESG", alt.Tooltip("Score", format=".5f")]
                )

            # Generate insight for ESG Scores
            esg_insight = get_graph_insight(esg_plot_df.to_dict(), "ESG Scores Timeline")
            col2.markdown(f"**Insight:** {esg_insight}")

        else:
            if line_metric == "Overall Score":
                line_metric = "Score"
                tone_df["WHO"] = company.title()
                ind_tone_df["WHO"] = "Industry Average"
                plot_df = pd.concat([tone_df, ind_tone_df]).reset_index(drop=True)
            else:
                df1 = df_company.groupby("DATE")[line_metric].mean(
                    ).reset_index()
                df2 = filter_on_date(df_data.groupby("DATE")[line_metric].mean(
                    ).reset_index(), start, end)
                df1["WHO"] = company.title()
                df2["WHO"] = "Industry Average"
                plot_df = pd.concat([df1, df2]).reset_index(drop=True)
            metric_chart = alt.Chart(plot_df, title="Trends Over Time"
                                   ).mark_line().encode(
                x=alt.X("yearmonthdate(DATE):O", title="DATE"),
                y=alt.Y(f"{line_metric}:Q", scale=alt.Scale(type="linear")),
                color=alt.Color("WHO", legend=None),
                strokeDash=alt.StrokeDash("WHO", sort=None,
                    legend=alt.Legend(
                        title=None, symbolType="stroke", symbolFillColor="gray",
                        symbolStrokeWidth=4, orient="top",
                        ),
                    ),
                tooltip=["DATE", alt.Tooltip(line_metric, format=".3f")]
                )
            
            # Generate insight for metric trend
            trend_insight = get_graph_insight(plot_df.to_dict(), f"{line_metric} Trend Analysis")
            col2.markdown(f"**Insight:** {trend_insight}")

        metric_chart = metric_chart.properties(
            height=340,
            width=200
        ).interactive()
        col2.altair_chart(metric_chart, use_container_width=True)

        ###### CHART: ESG RADAR ######
        col1, col2 = st.columns((1, 2))
        avg_esg = data["ESG"]
        avg_esg.rename(columns={"Unnamed: 0": "Type"}, inplace=True)
        avg_esg.replace({"T": "Overall", "E": "Environment",
                        "S": "Social", "G": "Governance"}, inplace=True)

        # Convert numeric columns to float, excluding the 'Type' column
        numeric_cols = avg_esg.columns.difference(['Type'])
        avg_esg[numeric_cols] = avg_esg[numeric_cols].apply(pd.to_numeric, errors='coerce')

        # Calculate industry average
        avg_esg["Industry Average"] = avg_esg[numeric_cols].mean(axis=1)

        radar_df = avg_esg[["Type", company, "Industry Average"]].melt("Type",
            value_name="score", var_name="entity")

        radar = px.line_polar(radar_df, r="score", theta="Type",
            color="entity", line_close=True, hover_name="Type",
            hover_data={"Type": True, "entity": True, "score": ":.2f"},
            color_discrete_map={"Industry Average": fuchsia, company: violet})
        radar.update_layout(template=None,
                          polar={
                              "radialaxis": {"showticklabels": False,
                                           "ticks": ""},
                              "angularaxis": {"showticklabels": False,
                                            "ticks": ""},
                          },
                          legend={"title": None, "yanchor": "middle",
                                 "orientation": "h"},
                          title={"text": "<b>ESG Scores</b>",
                                "x": 0.5, "y": 0.8875,
                                "xanchor": "center",
                                "yanchor": "top",
                                "font": {"family": "Futura", "size": 23}},
                          margin={"l": 5, "r": 5, "t": 0, "b": 0},
                          )
        radar.update_layout(showlegend=False)
        col1.plotly_chart(radar, use_container_width=True)

        # Generate insight for radar chart
        radar_insight = get_graph_insight(radar_df.to_dict(), "ESG Radar Comparison")
        col1.markdown(f"**Insight:** {radar_insight}")

        ###### CHART: DOCUMENT TONE DISTRIBUTION #####
        dist_chart = alt.Chart(df_company, title="Document Tone "
                             "Distribution").transform_density(
                density='Tone',
                as_=["Tone", "density"]
            ).mark_area(opacity=0.5,color="purple").encode(
                    x=alt.X('Tone:Q', scale=alt.Scale(domain=(-10, 10))),
                    y='density:Q',
                    tooltip=[alt.Tooltip("Tone", format=".3f"),
                            alt.Tooltip("density:Q", format=".4f")]
                ).properties(
                    height=325,
                ).configure_title(
                    dy=-20
                ).interactive()
        col2.markdown("### <br>", unsafe_allow_html=True)
        col2.altair_chart(dist_chart,use_container_width=True)

        # Generate insight for tone distribution
        tone_insight = get_graph_insight(df_company[['Tone']].to_dict(), "Document Tone Distribution")
        col2.markdown(f"**Insight:** {tone_insight}")

        ###### CHART: SCATTER OF ARTICLES OVER TIME #####
        scatter = alt.Chart(df_company, title="Article Tone").mark_circle().encode(
            x="NegativeTone:Q",
            y="PositiveTone:Q",
            size="WordCount:Q",
            color=alt.Color("Polarity:Q", scale=alt.Scale()),
            tooltip=[alt.Tooltip("Polarity", format=".3f"),
                    alt.Tooltip("NegativeTone", format=".3f"),
                    alt.Tooltip("PositiveTone", format=".3f"),
                    alt.Tooltip("DATE"),
                    alt.Tooltip("WordCount", format=",d"),
                    alt.Tooltip("SourceCommonName", title="Site")]
            ).properties(
                height=450
            ).interactive()
        st.altair_chart(scatter, use_container_width=True)

        # Generate insight for scatter plot
        scatter_insight = get_graph_insight(
            df_company[['NegativeTone', 'PositiveTone', 'WordCount', 'Polarity']].to_dict(),
            "Article Tone Scatter Plot"
        )
        st.markdown(f"**Insight:** {scatter_insight}")

        ###### NUMBER OF NEIGHBORS TO FIND #####
        company_df = df_conn[df_conn.company == company]
        
        # Get all similarity columns
        similar_org_cols = [col for col in df_conn.columns if 'similar_org' in col]
        similarity_cols = [col for col in df_conn.columns if 'similarity' in col]
        
        # Limit to the number of neighbors selected by the user
        similar_org_cols = similar_org_cols[:num_neighbors]
        similarity_cols = similarity_cols[:num_neighbors]
        
        # Get neighbors and their confidence scores
        if not company_df.empty and all(col in company_df.columns for col in similar_org_cols + similarity_cols):
            neighbors = company_df[similar_org_cols].iloc[0]
            neighbor_confidences = company_df[similarity_cols].iloc[0]

            ###### CHART: 3D EMBEDDING WITH NEIGHBORS ######
            st.markdown("---")
            color_f = lambda f: f"Company: {company.title()}" if f == company else (
                "Connected Company" if f in neighbors.values else "Other Company")
            embeddings["colorCode"] = embeddings.company.apply(color_f)
            point_colors = {
                f"Company: {company.title()}": violet,
                "Connected Company": fuchsia,
                "Other Company": "lightgrey"
            }
            
            fig_3d = px.scatter_3d(embeddings, x="0", y="1", z="2",
                                  color='colorCode',
                                  color_discrete_map=point_colors,
                                  opacity=0.4,
                                  hover_name="company",
                                  hover_data={c: False for c in embeddings.columns},
                                  )
            fig_3d.update_layout(
                legend={"orientation": "h",
                       "yanchor": "bottom",
                       "title": None},
                title={"text": "<b>Company Connections</b>",
                       "x": 0.5, "y": 0.9,
                       "xanchor": "center",
                       "yanchor": "top",
                       "font": {"family": "Futura", "size": 23}},
                scene={"xaxis": {"visible": False},
                      "yaxis": {"visible": False},
                      "zaxis": {"visible": False}},
                margin={"l": 0, "r": 0, "t": 0, "b": 0},
            )
            st.plotly_chart(fig_3d, use_container_width=True)

            # Generate insight for 3D embeddings
            embedding_insight = get_graph_insight(
                {
                    "company": company,
                    "neighbors": neighbors.to_dict(),
                    "connections": len(neighbors)
                },
                "3D Company Connections"
            )
            st.markdown(f"**Network Insight:** {embedding_insight}")

            ###### CHART: NEIGHBOR SIMILARITY ######
            st.markdown("---")
            neighbor_conf = pd.DataFrame({
                "Neighbor": neighbors.values,
                "Confidence": neighbor_confidences.values
            })
            
            conf_plot = alt.Chart(neighbor_conf, title="Connected Companies"
                                ).mark_bar().encode(
                x="Confidence:Q",
                y=alt.Y("Neighbor:N", sort="-x"),
                tooltip=["Neighbor", alt.Tooltip("Confidence", format=".3f")],
                color=alt.Color("Confidence:Q", scale=alt.Scale(), legend=None)
            ).properties(
                height=25 * num_neighbors + 100
            ).configure_axis(grid=False)
            
            st.altair_chart(conf_plot, use_container_width=True)

            # Generate insight for similarity scores
            similarity_insight = get_graph_insight(
                neighbor_conf.to_dict(),
                "Company Similarity Analysis"
            )
            st.markdown(f"**Similarity Insight:** {similarity_insight}")

        else:
            st.warning("No connection data available for this company.")

        # Add a final summary section
        st.markdown("---")
        st.markdown("### Overall Analysis Summary")
        
        # Generate overall summary using Groq
        summary_data = {
            "company": company,
            "total_articles": len(df_company),
            "avg_tone": df_company['Tone'].mean(),
            "avg_esg_scores": {
                "Environmental": esg_df[esg_df['ESG'] == 'Environment']['Score'].mean(),
                "Social": esg_df[esg_df['ESG'] == 'Social']['Score'].mean(),
                "Governance": esg_df[esg_df['ESG'] == 'Governance']['Score'].mean()
            },
            "date_range": f"{start} to {end}"
        }
        
        overall_insight = get_graph_insight(summary_data, "Overall Company Analysis")
        st.markdown(f"**Summary:** {overall_insight}")
if __name__ == "__main__":
    try:
        args = sys.argv
        if len(args) != 3:
            start_data = "jan1"
            end_data = "jan3"
        else:
            start_data = args[1]
            end_data = args[2]
        
        main(start_data, end_data)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    finally:
        alt.themes.enable("default")