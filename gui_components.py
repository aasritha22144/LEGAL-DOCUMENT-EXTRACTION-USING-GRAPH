import os
import tempfile
import logging
import platform
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict

from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget, QVBoxLayout, QWidget, QFileDialog,
    QTextEdit, QPushButton, QLabel, QComboBox, QMessageBox,
    QHBoxLayout, QListWidget, QListWidgetItem, QStatusBar, QApplication
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl, Qt
from PyQt5.QtGui import QFont

from data_loader import DataLoader
from legal_nlp import LegalNLPProcessor
from legal_graph import LegalGraphEngine

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.setup_ui()
        self.setup_connections()
        self.current_documents = []
        
        # Initialize graph engine with output directory
        self.graph_engine = LegalGraphEngine(output_dir="output")
        
        # Clean up old files on startup
        self.cleanup_old_files()
        
    def setup_ui(self):
        """Initialize all UI components"""
        self.setWindowTitle("Legal Document Extraction System")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central Widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Create components
        self.create_file_controls()
        self.create_tabs()
        
        # Initialize processors
        self.data_loader = DataLoader()
        self.nlp_processor = LegalNLPProcessor()
        
    def create_file_controls(self):
        """Create file loading controls"""
        file_controls = QHBoxLayout()
        
        self.btn_load_file = QPushButton("Load File")
        self.btn_load_file.setToolTip("Load a single legal document")
        self.btn_load_file.setFixedWidth(150)
        
        self.btn_load_dir = QPushButton("Load Directory")
        self.btn_load_dir.setToolTip("Load all documents from a folder")
        self.btn_load_dir.setFixedWidth(150)
        
        self.btn_open_output = QPushButton("Open Output Folder")
        self.btn_open_output.setToolTip("Open folder containing saved visualizations")
        self.btn_open_output.setFixedWidth(150)
        
        file_controls.addWidget(self.btn_load_file)
        file_controls.addWidget(self.btn_load_dir)
        file_controls.addWidget(self.btn_open_output)
        file_controls.addStretch()
        
        self.main_layout.addLayout(file_controls)
    
    def create_tabs(self):
        """Create main application tabs"""
        self.tabs = QTabWidget()
        
        # Documents Tab
        self.documents_tab = QWidget()
        self.setup_documents_tab()
        self.tabs.addTab(self.documents_tab, "Documents")
        
        # Analysis Tab
        self.analysis_tab = QWidget()
        self.setup_analysis_tab()
        self.tabs.addTab(self.analysis_tab, "Analysis")
        
        # Visualization Tab
        self.visualization_tab = QWidget()
        self.setup_visualization_tab()
        self.tabs.addTab(self.visualization_tab, "Knowledge Graph")
        
        self.main_layout.addWidget(self.tabs)
    
    def setup_documents_tab(self):
        """Setup documents tab components"""
        layout = QVBoxLayout()
        
        # Document List
        self.doc_list = QListWidget()
        self.doc_list.setMinimumHeight(150)
        self.doc_list.setFont(QFont("Arial", 10))
        
        # Document Content
        self.doc_content = QTextEdit()
        self.doc_content.setReadOnly(True)
        self.doc_content.setFont(QFont("Arial", 10))
        
        layout.addWidget(QLabel("Loaded Documents:"))
        layout.addWidget(self.doc_list)
        layout.addWidget(QLabel("Document Content:"))
        layout.addWidget(self.doc_content)
        self.documents_tab.setLayout(layout)
    
    def setup_analysis_tab(self):
        """Setup analysis tab components"""
        layout = QVBoxLayout()
        
        # Analysis Type
        self.analysis_type = QComboBox()
        self.analysis_type.addItems([
            "Named Entity Recognition",
            "Relation Extraction",
            "Document Summarization"
        ])
        self.analysis_type.setFont(QFont("Arial", 10))
        
        # Run Analysis Button
        self.btn_run_analysis = QPushButton("Run Analysis")
        self.btn_run_analysis.setEnabled(False)
        self.btn_run_analysis.setFixedWidth(150)
        
        # Results Display
        self.analysis_results = QTextEdit()
        self.analysis_results.setReadOnly(True)
        self.analysis_results.setFont(QFont("Arial", 10))
        
        layout.addWidget(QLabel("Analysis Type:"))
        layout.addWidget(self.analysis_type)
        layout.addWidget(self.btn_run_analysis)
        layout.addWidget(QLabel("Results:"))
        layout.addWidget(self.analysis_results)
        self.analysis_tab.setLayout(layout)
    
    def setup_visualization_tab(self):
        """Setup visualization tab components"""
        layout = QVBoxLayout()
        
        # Generate Visualization Button
        self.btn_generate_viz = QPushButton("Generate Visualization")
        self.btn_generate_viz.setEnabled(False)
        self.btn_generate_viz.setFixedWidth(200)
        
        # Web View
        self.web_view = QWebEngineView()
        self.web_view.setHtml("""
            <html>
                <body style="background-color:#f0f0f0;">
                    <h2 style="color:#666;text-align:center;">
                        Visualization will appear here
                    </h2>
                </body>
            </html>
        """)
        
        layout.addWidget(self.btn_generate_viz)
        layout.addWidget(self.web_view)
        self.visualization_tab.setLayout(layout)
    
    def setup_connections(self):
        """Connect signals to slots"""
        self.btn_load_file.clicked.connect(self.load_file)
        self.btn_load_dir.clicked.connect(self.load_directory)
        self.btn_run_analysis.clicked.connect(self.run_analysis)
        self.btn_generate_viz.clicked.connect(self.generate_visualization)
        self.btn_open_output.clicked.connect(self.open_output_folder)
        self.doc_list.itemClicked.connect(self.show_document_content)
    
    def load_file(self):
        """Handle file loading with full error handling"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Open Legal Document",
                str(Path.home() / "Documents"),
                "Legal Documents (*.pdf *.docx *.txt);;All Files (*)"
            )
            
            if not file_path:  # User cancelled
                return
                
            self.status_bar.showMessage("Loading file...")
            QApplication.processEvents()
            
            doc = self.data_loader.load_file(file_path)
            if not doc:
                raise ValueError("Failed to load document content")
                
            self.current_documents = [doc]
            self.update_document_list()
            self.enable_analysis_controls(True)
            self.status_bar.showMessage(f"Loaded: {doc['filename']}", 3000)
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load file:\n\n{str(e)}"
            )
            self.status_bar.showMessage("File load failed", 3000)
            self.logger.error(f"File load error: {str(e)}", exc_info=True)
    
    def load_directory(self):
        """Handle directory loading with full error handling"""
        try:
            dir_path = QFileDialog.getExistingDirectory(
                self,
                "Select Documents Directory",
                str(Path.home() / "Documents")
            )
            
            if not dir_path:  # User cancelled
                return
                
            self.status_bar.showMessage("Loading directory...")
            QApplication.processEvents()
            
            docs = self.data_loader.load_directory(dir_path)
            if not docs:
                raise ValueError("No supported documents found")
                
            self.current_documents = docs
            self.update_document_list()
            self.enable_analysis_controls(True)
            self.status_bar.showMessage(f"Loaded {len(docs)} documents", 3000)
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load directory:\n\n{str(e)}"
            )
            self.status_bar.showMessage("Directory load failed", 3000)
            self.logger.error(f"Directory load error: {str(e)}", exc_info=True)
    
    def update_document_list(self):
        """Update the documents list widget"""
        self.doc_list.clear()
        for doc in self.current_documents:
            item = QListWidgetItem(doc['filename'])
            item.setData(Qt.UserRole, doc)
            self.doc_list.addItem(item)
        self.doc_list.setCurrentRow(0)
        self.show_document_content(self.doc_list.currentItem())
    
    def show_document_content(self, item):
        """Display selected document content"""
        if item and (doc := item.data(Qt.UserRole)):
            self.doc_content.setPlainText(doc['text'])
    
    def enable_analysis_controls(self, enabled):
        """Enable/disable analysis controls"""
        self.btn_run_analysis.setEnabled(enabled)
        self.btn_generate_viz.setEnabled(enabled)
    
    def run_analysis(self):
        """Execute document analysis with full error handling"""
        if not self.current_documents:
            QMessageBox.warning(self, "Error", "No documents loaded")
            return
        
        try:
            doc = self.current_documents[0]
            analysis_type = self.analysis_type.currentText()
            
            self.status_bar.showMessage("Running analysis...")
            QApplication.processEvents()
            
            if analysis_type == "Named Entity Recognition":
                entities = self.nlp_processor.extract_entities(doc['text'])
                result = "Named Entities:\n" + "\n".join(
                    f"- {e['text']} ({e['type']})" for e in entities
                ) if entities else "No named entities found"
            
            elif analysis_type == "Relation Extraction":
                relations = self.nlp_processor.extract_relations(doc['text'])
                result = "Relations:\n" + "\n".join(
                    f"- {r['source']} → {r['target']} ({r['relation']})\n  Context: {r['sentence']}"
                    for r in relations
                ) if relations else "No relations found"
            
            else:  # Summarization
                result = "Summary:\n" + self.nlp_processor.summarize(doc['text'])
            
            self.analysis_results.setPlainText(result)
            self.status_bar.showMessage("Analysis completed", 3000)
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Analysis Error",
                f"Analysis failed:\n\n{str(e)}"
            )
            self.status_bar.showMessage("Analysis failed", 3000)
            self.logger.error(f"Analysis error: {str(e)}", exc_info=True)
    
    def generate_visualization(self):
        """Generate knowledge graph visualization with full error handling"""
        if not self.current_documents:
            QMessageBox.warning(self, "Error", "No documents loaded")
            return
        
        try:
            doc = self.current_documents[0]
            text = doc['text']
            
            # Show loading message
            self.set_visualization_loading()
            QApplication.processEvents()
            
            # Process text
            entities = self.nlp_processor.extract_entities(text)
            relations = self.nlp_processor.extract_relations(text)
            
            if not entities and not relations:
                self.show_visualization_error("No entities or relations found")
                self.status_bar.showMessage("No data to visualize", 3000)
                return
            
            # Build and visualize graph
            if not self.graph_engine.build_graph(entities, relations):
                self.show_visualization_error("Failed to build knowledge graph")
                self.status_bar.showMessage("Graph build failed", 3000)
                return
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"graph_{timestamp}.png"
            
            if (graph_path := self.graph_engine.visualize(filename)):
                self.display_visualization(graph_path)
                self.status_bar.showMessage(f"Visualization saved to {graph_path}", 5000)
            else:
                self.show_visualization_error("Failed to generate visualization")
                self.status_bar.showMessage("Visualization failed", 3000)
                
        except Exception as e:
            self.show_visualization_error(f"Error: {str(e)}")
            self.status_bar.showMessage("Visualization error", 3000)
            self.logger.error(f"Visualization error: {str(e)}", exc_info=True)
    
    def open_output_folder(self):
        """Open the output folder in system file explorer"""
        try:
            path = str(self.graph_engine.output_dir.resolve())
            
            if platform.system() == "Windows":
                os.startfile(path)
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
                
            self.status_bar.showMessage(f"Opened output folder: {path}", 3000)
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to open output folder:\n\n{str(e)}"
            )
            self.status_bar.showMessage("Failed to open output folder", 3000)
            self.logger.error(f"Failed to open output folder: {str(e)}", exc_info=True)
    
    def cleanup_old_files(self, days_to_keep=7):
        """Clean up old visualization files"""
        try:
            cutoff = datetime.now() - timedelta(days=days_to_keep)
            deleted_files = 0
            
            for file in self.graph_engine.output_dir.glob("*.png"):
                if datetime.fromtimestamp(file.stat().st_mtime) < cutoff:
                    file.unlink()
                    deleted_files += 1
                    self.logger.info(f"Deleted old file: {file}")
            
            if deleted_files > 0:
                self.logger.info(f"Cleaned up {deleted_files} old visualization files")
                
        except Exception as e:
            self.logger.error(f"Failed to clean up old files: {str(e)}", exc_info=True)
    
    def set_visualization_loading(self):
        """Show loading state in visualization tab"""
        self.web_view.setHtml("""
            <html>
                <body style="background-color:#f0f0f0;">
                    <h2 style="color:#666;text-align:center;">
                        Generating visualization...
                    </h2>
                </body>
            </html>
        """)
    
    def show_visualization_error(self, message):
        """Show error message in visualization tab"""
        self.web_view.setHtml(f"""
            <html>
                <body style="background-color:#f0f0f0;">
                    <h2 style="color:red;text-align:center;">
                        {message}
                    </h2>
                </body>
            </html>
        """)
    
    def display_visualization(self, image_path):
        """Display the generated visualization"""
        self.web_view.setHtml(f"""
            <html>
                <body style="background-color:#f0f0f0;text-align:center;">
                    <img src="{image_path}" style="max-width:100%; height:auto;">
                    <p style="color:#666;">Knowledge Graph Visualization</p>
                </body>
            </html>
        """)