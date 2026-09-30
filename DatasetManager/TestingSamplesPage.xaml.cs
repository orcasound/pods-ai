using System;
using System.Collections.Generic;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace DatasetManager
{
    /// <summary>
    /// Interaction logic for TestingSamplesPage.xaml
    /// </summary>
    public partial class TestingSamplesPage : Page
    {
        private readonly string _category;
        private List<SampleRecord> _samples;
        public List<SampleRecord> Samples => _samples.Where(s => s.Category == _category).ToList();

        private void Back_Click(object sender, RoutedEventArgs e)
        {
            NavigationService?.GoBack();
        }

        public TestingSamplesPage(List<SampleRecord> samples, string category)
        {
            InitializeComponent();
            _samples = samples;
            _category = category;
            DataContext = this;
        }
    }
}
