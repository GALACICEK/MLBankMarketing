import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class EDA:

    my_colormap= ['#00876c','#85b96f','#f7e382','#f19452','#fdc4b6','#ea7070','#d43d51','#6495ED','#4169E1','#385a7c']

    def __init__(self, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input 'df' must be a DataFrame.")
        self.df = df.copy()

# Categorical Functions---------------------------------------------
    
    def find_categorical_unique_features(self, target_column=None):
        """
        Find categorical features and their unique category counts in the DataFrame.

        Parameters:
        - target_column (str): The target column name to exclude from categorical features. Default is None.

        Returns:
        result_df (pd.DataFrame): A DataFrame containing categorical features and their unique category counts.
        """
        categorical_features = [feature for feature in self.df.columns if self.df[feature].dtype == 'O' and feature != target_column]

        result_df = pd.DataFrame(columns=['Feature', 'Unique Categories'])

        for feature in categorical_features:
            unique_categories = self.df[feature].nunique()
            result_df = pd.concat([result_df, pd.DataFrame({'Feature': [feature], 'Unique Categories': [unique_categories]})], ignore_index=True)


        return result_df

    def find_categorical_describle(self,categ_columns):
        """
        Generate descriptive statistics for the specified categorical columns in the DataFrame.

        Parameters:
        - categ_columns (dict): A dictionary where keys are column names and values are categorical data.

        Returns:
        pd.DataFrame: A DataFrame containing descriptive statistics for the specified categorical columns.
        """
        categ_df = pd.DataFrame(categ_columns)
        return categ_df.describe().T
    
    def countplot_categorical(self,categ_features,hue_column=None):
        """
        Create countplots to visualize the distribution of categorical features.

        Parameters:
        - categ_features (list): A list of categorical feature names to be plotted.
        - hue_column (str, optional): The name of the column to be used as hue (e.g., 'yes' or 'no'). Default is None.

        Returns:
        None
        """
        
        color = ['#00876c'] if hue_column is None else ['#00876c','#85b96f']
        plt.figure(figsize=(20,50))
        for idx, categ_feature in enumerate(categ_features):
            # Created subplot for each features
            ax = plt.subplot(len(categ_features) // 2 + 1, 2, idx + 1)
            sns.countplot(data = self.df,
                          x = categ_feature,
                          ax=ax,
                          hue = hue_column ,
                          palette=color)
            
            #Annotate the height values on the top of each bar in a bar plot
            for p in ax.patches:
                ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                            textcoords='offset points')

            # Set labels 
            ax.set_xticks(range(len(self.df[categ_feature].unique())))
            ax.set_xticklabels(self.df[categ_feature].unique(), rotation=45)
        
            ax.set_xlabel(categ_feature, fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title(f'{categ_feature} Count Distribution', fontsize=15)
            plt.tick_params(labelsize=12)

            plt.tight_layout()
            plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.show()

        return
    
    def boxplot_categorical_relationship(self,categ_features,numeric_feature,hue_column=None):
        """
        Create boxplots to visualize the relationship between categorical features and a numeric feature.
            Show median, outliers (Q1, Q3) variables on boxplot

        Parameters:
        - categ_features (list): A list of categorical feature names to be plotted.
        - numeric_feature (str): The name of the numeric feature.
        - hue_column (str, optional): The name of the column to be used as hue (e.g., 'yes' or 'no'). Default is None.

        Returns:
        None
        """
        color = ['#4dbedf'] if hue_column is None else ['#fdc4b6','#ea7070']
        legend = False if hue_column is None else True

        plt.figure(figsize=(20,50))
        for idx, categ_feature in enumerate(categ_features):
            ax = plt.subplot(len(categ_features) // 2 + 1, 2, idx + 1)
            sns.boxplot(data = self.df,
                        x = categ_feature,
                        y= numeric_feature, 
                        ax=ax,
                        showfliers=True,  # Show Outliers(Q1 and Q3)
                        hue = hue_column,
                        legend=legend,
                        palette=color)
            
            # This part for show median , q1, q3 values on graph
            if hue_column is None :   

                # Show median ve Q1, Q3 values on graph
                med = self.df.groupby(categ_feature)[numeric_feature].median().to_dict()
                q1 = self.df.groupby(categ_feature)[numeric_feature].quantile(0.25).to_dict()
                q3 = self.df.groupby(categ_feature)[numeric_feature].quantile(0.75).to_dict()


                for tick, label in enumerate(ax.get_xticklabels()):
                    category = label.get_text()

                    median_value = med.get(category,None)
                    if median_value is not None:
                        ax.text(tick, median_value + 0.5, f'Median: {median_value:.2f}', 
                                ha='center', va='center', color='k', fontsize=8)
                        
                    q1_value = q1.get(category,None)
                    if q1_value is not None:
                        ax.text(tick, q1_value + 0.5, f"Q1: {q1_value:.2f}", 
                                ha='center', va='center', color='k', fontsize=8)
                        
                    q3_value = q3.get(category,None)
                    if q3_value is not None:
                        ax.text(tick, q3_value + 0.5, f"Q3: {q3_value:.2f}",
                                 ha='center', va='center', color='k', fontsize=8)
            
            else:
                # Show median ve Q1, Q3 values on graph with hue column
                med = self.df.groupby([categ_feature,hue_column])[numeric_feature].median().to_dict()
                q1 = self.df.groupby([categ_feature,hue_column])[numeric_feature].quantile(0.25).to_dict()
                q3 = self.df.groupby([categ_feature,hue_column])[numeric_feature].quantile(0.75).to_dict()
                
                for tick, label in enumerate(ax.get_xticklabels()):
                    for hue_tick, hue_label in enumerate(ax.get_legend().get_texts()):
                        category = label.get_text()
                        hue_category = hue_label.get_text()

                        # Median value for hue
                        median_value = med.get((category, hue_category), None)
                        if median_value is not None:
                            ax.text(tick + hue_tick * 0.2, median_value + 0.5, f"Median: {median_value:.2f}",
                                    ha='center', va='center', color='k', fontsize=8)
                        
                        # Q1 value for hue
                        q1_value = q1.get((category, hue_category), None)
                        if q1_value is not None:
                            ax.text(tick + hue_tick * 0.2, q1_value + 0.5, f"Q1: {q1_value:.2f}",
                                    ha='center', va='center', color='k', fontsize=8)

                        # Q3 value for hue
                        q3_value = q3.get((category, hue_category), None)
                        if q3_value is not None:
                            ax.text(tick + hue_tick * 0.2, q3_value + 0.5, f"Q3: {q3_value:.2f}",
                                    ha='center', va='center', color='k', fontsize=8)


            # Set labels
            ax.set_xticks(range(len(self.df[categ_feature].unique())))
            ax.set_xticklabels(self.df[categ_feature].unique(), rotation=45)

            ax.set_xlabel(categ_feature, fontsize=12)
            ax.set_ylabel(numeric_feature, fontsize=12)
            ax.set_title(f'{categ_feature} and {numeric_feature} relationship distribution', fontsize=15)
            ax.tick_params(labelsize=12)

            plt.tight_layout()
            plt.subplots_adjust(wspace=0.5, hspace=0.5) 
        plt.show()
    
        return
    

    
    def find_corr_heatmap(self,df_encoding):
        """
        Generates and displays a heatmap of Spearman rank correlation coefficients for the dataset.

        This function calculates the Spearman rank correlation coefficients for the numeric columns in
        the dataset and creates a triangular heatmap to visualize the relationships between variables.

        Returns:
        None
        """

        if not isinstance(df_encoding, pd.DataFrame):
            raise ValueError("Input 'df' must be include a numeric values DataFrame.")
        

        plt.figure(figsize=(20,20))
        triangle_mask = np.triu(np.ones_like(df_encoding.corr('spearman')))
        heatmap = sns.heatmap(data= df_encoding.corr('spearman'),
                              mask=triangle_mask,
                              vmin=-1,
                              vmax=1,
                              annot=True,
                              cmap=self.my_colormap)
        heatmap.set_title("Triangle Correlation Heatmap",
                          fontdict={'fontsize':12}, pad=20)
        plt.show()
        
        return
    
# Numeric Functions---------------------------------------------
    
    def find_numeric_features_list(self, target_column=None):
        """
        Find numeric features in the DataFrame.

        Parameters:
        - target_column (str): The target column name to exclude from numeric features. Default is None.

        Returns:
        list: A list containing numeric features.
        """
        numeric_features = [feature for feature in self.df.columns if self.df[feature].dtype != 'O' and feature != target_column]
        return numeric_features
    
    def find_numeric_describle(self,numeric_columns):
        """
        Generate descriptive statistics for the specified numeric columns in the DataFrame.
            

        Parameters:
        - numeric_columns (list): A list of column names containing numeric data.

        Returns:
        pd.DataFrame: A DataFrame containing descriptive statistics for the specified numeric columns.
        """
        numeric_df = pd.DataFrame(numeric_columns)
        return numeric_columns.describe().T
    
    def histplot_numeric(self,numeric_features,hue_column=None,figsize=(15, 5)):
        """
        Create histogram plots to visualize the distribution of numeric features.
        Show feature vertical median line on histplot.

        Parameters:
        - numeric_features (list): A list of numeric feature names to be plotted.
        - hue_column (str, optional): The name of the column to be used as hue (e.g., 'yes' or 'no'). Default is None.
        - figsize (tuple, optional): Figure size in inches. Default is (15, 5).

        Returns:
        None
        """

        # Determine color based on the presence of hue_column
        color = '#385a7c' if hue_column is None else ['#6495ED','#4169E1']

        plt.figure(figsize=figsize)
        for idx, numeric_feature in enumerate(numeric_features):
            ax = plt.subplot((len(numeric_features) + 1) // 2 , min(2, len(numeric_features)), idx + 1)
            sns.histplot(data = self.df,
                          x = numeric_feature,
                          kde=True,
                          ax=ax,
                          label = numeric_feature,
                          hue = hue_column ,
                          edgecolor='black',
                          color=color)
            
            plt.axvline(x=self.df[numeric_feature].mean(),
                        color = 'k',
                        linestyle ='--',
                        label = 'Mean {}'.format(round(self.df[numeric_feature].mean(),2))
                        )
            plt.legend()


            # Set labels
            ax.set_xlabel(numeric_feature, fontsize=12)
            ax.set_xlim(self.df[numeric_feature].min()-5, self.df[numeric_feature].max()+5)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title(f'{numeric_feature} Distribution', fontsize=15)
            ax.tick_params(labelsize=12)

            plt.tight_layout()
            
        plt.show()
        return






