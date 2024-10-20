import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import tkinter as tk
from tkinter import filedialog

nltk.download('vader_lexicon')

def load_file_dialog():
    root = tk.Tk()
    root.withdraw()  
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    return file_path

class DataAnalysis:
    def __init__(self, filename):
        self.filename = filename
        self.df = self.load_dataset()

    def load_dataset(self):
        try:
            df = pd.read_csv(self.filename)
            print(f"Dataset '{self.filename}' loaded successfully!")
            return df
        except FileNotFoundError:
            print(f"File '{self.filename}' not found.")
            return None

    def main_menu(self):
        while True:
            print("\nHow do you want to analyze your data?")
            print("1. Plot variable distribution")
            print("2. Conduct ANOVA")
            print("3. Conduct t-Test")
            print("4. Conduct chi-Square")
            print("5. Conduct Regression")
            print("6. Conduct Sentiment Analysis")
            print("7. Quit")

            choice = input("Enter your choice (1-7): ")

            if choice == '1':
                self.plot_variable_distribution()
            elif choice == '2':
                self.perform_anova()
            elif choice == '3':
                self.perform_t_test()
            elif choice == '4':
                self.perform_chi_square()
            elif choice == '5':
                self.perform_regression()
            elif choice == '6':
                self.perform_sentiment_analysis()
            elif choice == '7':
                print("Exiting the program. Goodbye!")
                break
            else:
                print("Invalid choice. Please select again.")

    def plot_variable_distribution(self):
        while True:
            print("\nAvailable columns for plotting:")
            for i, col in enumerate(self.df.columns):
                print(f"{i+1}. {col}")
            print(f"{len(self.df.columns)+1}. BACK")
            print(f"{len(self.df.columns)+2}. QUIT")
            choice = int(input("Choose a variable to plot or go BACK/QUIT: "))
            
            if choice == len(self.df.columns) + 1:
                return  
            elif choice == len(self.df.columns) + 2:
                print("Exiting the program. Goodbye!")
                exit()
            elif 1 <= choice <= len(self.df.columns):
                column = self.df.columns[choice - 1]
                self.plot_distribution(column)
            else:
                print("Invalid choice. Please select again.")

    def plot_distribution(self, column):
        if pd.api.types.is_numeric_dtype(self.df[column]):
            sns.histplot(self.df[column], kde=True)
            plt.title(f'Distribution of {column}')
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.show()
        else:
            sns.countplot(x=column, data=self.df)
            plt.title(f'Distribution of {column}')
            plt.xlabel(column)
            plt.ylabel("Count")
            plt.show()

    def perform_anova(self):
        while True:
            print("\nSelect continuous and categorical variables for ANOVA:")
        
            all_vars = self.df.columns.tolist()

        
            for i, col in enumerate(all_vars):
                print(f"{i+1}. {col}")
            print(f"{len(all_vars)+1}. BACK")
            print(f"{len(all_vars)+2}. QUIT")

            continuous_var = int(input("Choose a continuous variable or go BACK/QUIT: "))

            if continuous_var == len(all_vars) + 1:
                return  
            elif continuous_var == len(all_vars) + 2:
                print("Exiting the program. Goodbye!")
                exit()

            selected_continuous_var = all_vars[continuous_var - 1]

       
            print(f"\nChoose a categorical variable from: {all_vars}: ")
            categorical_var = input()

            if categorical_var in all_vars:
                self.check_normality_and_anova(selected_continuous_var, categorical_var)
            else:
                print("Invalid selection. Try again.")


    def check_normality_and_anova(self, continuous_var, categorical_var):
        stat, p = stats.shapiro(self.df[continuous_var])
        print(f"Shapiro-Wilk test: p-value = {p}")
        if p < 0.05:
            print(f"‘{continuous_var}’ is not normally distributed, performing Kruskal-Wallis test...")
            self.perform_kruskal_wallis(continuous_var, categorical_var)
        else:
            print(f"‘{continuous_var}’ is normally distributed, performing ANOVA...")
            model = sm.formula.ols(f'{continuous_var} ~ C({categorical_var})', data=self.df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            print(anova_table)
            self.plot_anova(continuous_var, categorical_var)

    def perform_kruskal_wallis(self, continuous_var, categorical_var):
        groups = [group[continuous_var].dropna() for name, group in self.df.groupby(categorical_var)]
        stat, p = stats.kruskal(*groups)
        print(f"Kruskal-Wallis Statistic: {stat}, p-value: {p}")
        self.plot_anova(continuous_var, categorical_var)

    def plot_anova(self, continuous_var, categorical_var):
        sns.boxplot(x=categorical_var, y=continuous_var, data=self.df)
        plt.title(f'Boxplot of {continuous_var} by {categorical_var}')
        plt.xlabel(categorical_var)
        plt.ylabel(continuous_var)
        plt.show()

    def perform_t_test(self):
        while True:
            print("\nSelect two continuous variables for t-Test:")
            continuous_vars = [col for col in self.df.columns if pd.api.types.is_numeric_dtype(self.df[col])]
            
            for i, col in enumerate(continuous_vars):
                print(f"{i+1}. {col}")
            print(f"{len(continuous_vars)+1}. BACK")
            print(f"{len(continuous_vars)+2}. QUIT")
            
            var1 = int(input("Choose the first variable or go BACK/QUIT: "))
            
            if var1 == len(continuous_vars) + 1:
                return  
            elif var1 == len(continuous_vars) + 2:
                print("Exiting the program. Goodbye!")
                exit()

            print("\nSelect the second variable:")
            for i, col in enumerate(continuous_vars):
                print(f"{i+1}. {col}")
                
            var2 = int(input("Choose the second variable: "))

            if var2 <= len(continuous_vars):
                self.perform_t_test_logic(continuous_vars[var1 - 1], continuous_vars[var2 - 1])
            else:
                print("Invalid selection. Try again.")

    def perform_t_test_logic(self, var1, var2):
        t_stat, p_value = stats.ttest_ind(self.df[var1].dropna(), self.df[var2].dropna())
        print(f"t-Test result: t-statistic = {t_stat}, p-value = {p_value}")
        if p_value < 0.05:
            print("Statistically significant difference between the two groups.")
        else:
            print("No statistically significant difference between the two groups.")
        self.plot_t_test(var1, var2)

    def plot_t_test(self, var1, var2):
        sns.kdeplot(self.df[var1], label=var1, shade=True)
        sns.kdeplot(self.df[var2], label=var2, shade=True)
        plt.title(f'KDE Plot of {var1} and {var2}')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.show()

    def perform_chi_square(self):
        while True:
            print("\nSelect two variables for chi-Square test:")
            all_vars = self.df.columns.tolist()
        
            for i, col in enumerate(all_vars):
                print(f"{i+1}. {col}")
            print(f"{len(all_vars)+1}. BACK")
            print(f"{len(all_vars)+2}. QUIT")
        
            var1 = int(input("Choose the first variable or go BACK/QUIT: "))
        
            if var1 == len(all_vars) + 1:
                return
            elif var1 == len(all_vars) + 2:
                print("Exiting the program. Goodbye!")
                exit()

            var2 = int(input("Choose the second variable: "))

            if var2 <= len(all_vars):
                self.perform_chi_square_logic(all_vars[var1 - 1], all_vars[var2 - 1])
            else:
                print("Invalid selection. Try again.")

    def perform_chi_square_logic(self, var1, var2):
        contingency_table = pd.crosstab(self.df[var1], self.df[var2])
        chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency_table)
        print(f"Chi-Square result: chi2 = {chi2_stat}, p-value = {p_val}")
        if p_val < 0.05:
            print("Statistically significant association between the variables.")
        else:
            print("No statistically significant association between the variables.")
        self.plot_chi_square(var1, var2)

    def plot_chi_square(self, var1, var2):
        contingency_table = pd.crosstab(self.df[var1], self.df[var2])
        sns.heatmap(contingency_table, annot=False, fmt="d", cmap="YlGnBu")
        plt.title(f'Heatmap of {var1} vs {var2}')
        plt.xlabel(var2)
        plt.ylabel(var1)
        plt.show()

    def perform_regression(self):
        while True:
            print("\nSelect dependent and independent variables for regression:")
            continuous_vars = [col for col in self.df.columns if pd.api.types.is_numeric_dtype(self.df[col])]
            
            for i, col in enumerate(continuous_vars):
                print(f"{i+1}. {col}")
            print(f"{len(continuous_vars)+1}. BACK")
            print(f"{len(continuous_vars)+2}. QUIT")
            
            dependent_var = int(input("Choose the dependent variable or go BACK/QUIT: "))
            
            if dependent_var == len(continuous_vars) + 1:
                return
            elif dependent_var == len(continuous_vars) + 2:
                print("Exiting the program. Goodbye!")
                exit()

            print("\nSelect the independent variable:")
            for i, col in enumerate(continuous_vars):
                print(f"{i+1}. {col}")

            independent_var = int(input("Choose the independent variable: "))

            if independent_var <= len(continuous_vars):
                self.perform_regression_logic(continuous_vars[dependent_var - 1], continuous_vars[independent_var - 1])
            else:
                print("Invalid selection. Try again.")

    def perform_regression_logic(self, dependent_var, independent_var):
        X = sm.add_constant(self.df[independent_var])
        model = sm.OLS(self.df[dependent_var], X).fit()
        print(model.summary())
        self.plot_regression(dependent_var, independent_var)

    def plot_regression(self, dependent_var, independent_var):
        sns.regplot(x=independent_var, y=dependent_var, data=self.df, ci=None)
        plt.title(f'Regression Plot: {dependent_var} vs {independent_var}')
        plt.xlabel(independent_var)
        plt.ylabel(dependent_var)
        plt.show()

    def perform_sentiment_analysis(self):
        while True:
            print("\nPerforming sentiment analysis:")
            text_columns = self.df.select_dtypes(include=['object']).columns
            
            for i, col in enumerate(text_columns):
                print(f"{i+1}. {col}")
            print(f"{len(text_columns)+1}. BACK")
            print(f"{len(text_columns)+2}. QUIT")
            
            text_col = int(input("Choose a text column or go BACK/QUIT: "))
            
            if text_col == len(text_columns) + 1:
                return
            elif text_col == len(text_columns) + 2:
                print("Exiting the program. Goodbye!")
                exit()

            if text_col <= len(text_columns):
                self.perform_sentiment_analysis_logic(text_columns[text_col - 1])
            else:
                print("Invalid selection. Try again.")

    def perform_sentiment_analysis_logic(self, text_col):
        sid = SentimentIntensityAnalyzer()
        self.df['sentiment_score'] = self.df[text_col].apply(lambda x: sid.polarity_scores(str(x))['compound'])
        self.df['sentiment'] = self.df['sentiment_score'].apply(lambda score: 'positive' if score >= 0.05 else 'negative' if score <= -0.05 else 'neutral')
        print(self.df[[text_col, 'sentiment_score', 'sentiment']].head())
        self.plot_sentiment_analysis(text_col)

    def plot_sentiment_analysis(self, text_col):
        sns.countplot(x='sentiment', data=self.df)
        plt.title(f'Sentiment Analysis of {text_col}')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.show()

def main():
    filename = load_file_dialog()
    if filename:
        analysis = DataAnalysis(filename)
        if analysis.df is not None:
            analysis.main_menu()

if __name__ == '__main__':
    main()
