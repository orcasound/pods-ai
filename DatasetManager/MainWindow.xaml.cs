// Copyright (c) PODS-AI contributors
// SPDX-License-Identifier: MIT
using System.Windows;

namespace DatasetManager
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
            MainFrame.Navigate(new OverviewPage());
        }
    }
}