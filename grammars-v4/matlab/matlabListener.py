# Generated from matlab.g4 by ANTLR 4.13.2
from antlr4 import *
if "." in __name__:
    from .matlabParser import matlabParser
else:
    from matlabParser import matlabParser

# This class defines a complete listener for a parse tree produced by matlabParser.
class matlabListener(ParseTreeListener):

    # Enter a parse tree produced by matlabParser#atom_boolean.
    def enterAtom_boolean(self, ctx:matlabParser.Atom_booleanContext):
        pass

    # Exit a parse tree produced by matlabParser#atom_boolean.
    def exitAtom_boolean(self, ctx:matlabParser.Atom_booleanContext):
        pass


    # Enter a parse tree produced by matlabParser#atom_empty_array.
    def enterAtom_empty_array(self, ctx:matlabParser.Atom_empty_arrayContext):
        pass

    # Exit a parse tree produced by matlabParser#atom_empty_array.
    def exitAtom_empty_array(self, ctx:matlabParser.Atom_empty_arrayContext):
        pass


    # Enter a parse tree produced by matlabParser#atom_empty_cell.
    def enterAtom_empty_cell(self, ctx:matlabParser.Atom_empty_cellContext):
        pass

    # Exit a parse tree produced by matlabParser#atom_empty_cell.
    def exitAtom_empty_cell(self, ctx:matlabParser.Atom_empty_cellContext):
        pass


    # Enter a parse tree produced by matlabParser#atom_end.
    def enterAtom_end(self, ctx:matlabParser.Atom_endContext):
        pass

    # Exit a parse tree produced by matlabParser#atom_end.
    def exitAtom_end(self, ctx:matlabParser.Atom_endContext):
        pass


    # Enter a parse tree produced by matlabParser#atom_float.
    def enterAtom_float(self, ctx:matlabParser.Atom_floatContext):
        pass

    # Exit a parse tree produced by matlabParser#atom_float.
    def exitAtom_float(self, ctx:matlabParser.Atom_floatContext):
        pass


    # Enter a parse tree produced by matlabParser#atom_imaginary.
    def enterAtom_imaginary(self, ctx:matlabParser.Atom_imaginaryContext):
        pass

    # Exit a parse tree produced by matlabParser#atom_imaginary.
    def exitAtom_imaginary(self, ctx:matlabParser.Atom_imaginaryContext):
        pass


    # Enter a parse tree produced by matlabParser#atom_index_all.
    def enterAtom_index_all(self, ctx:matlabParser.Atom_index_allContext):
        pass

    # Exit a parse tree produced by matlabParser#atom_index_all.
    def exitAtom_index_all(self, ctx:matlabParser.Atom_index_allContext):
        pass


    # Enter a parse tree produced by matlabParser#atom_integer.
    def enterAtom_integer(self, ctx:matlabParser.Atom_integerContext):
        pass

    # Exit a parse tree produced by matlabParser#atom_integer.
    def exitAtom_integer(self, ctx:matlabParser.Atom_integerContext):
        pass


    # Enter a parse tree produced by matlabParser#atom_string.
    def enterAtom_string(self, ctx:matlabParser.Atom_stringContext):
        pass

    # Exit a parse tree produced by matlabParser#atom_string.
    def exitAtom_string(self, ctx:matlabParser.Atom_stringContext):
        pass


    # Enter a parse tree produced by matlabParser#atom_meta.
    def enterAtom_meta(self, ctx:matlabParser.Atom_metaContext):
        pass

    # Exit a parse tree produced by matlabParser#atom_meta.
    def exitAtom_meta(self, ctx:matlabParser.Atom_metaContext):
        pass


    # Enter a parse tree produced by matlabParser#atom_var.
    def enterAtom_var(self, ctx:matlabParser.Atom_varContext):
        pass

    # Exit a parse tree produced by matlabParser#atom_var.
    def exitAtom_var(self, ctx:matlabParser.Atom_varContext):
        pass


    # Enter a parse tree produced by matlabParser#matlab_file.
    def enterMatlab_file(self, ctx:matlabParser.Matlab_fileContext):
        pass

    # Exit a parse tree produced by matlabParser#matlab_file.
    def exitMatlab_file(self, ctx:matlabParser.Matlab_fileContext):
        pass


    # Enter a parse tree produced by matlabParser#def_function.
    def enterDef_function(self, ctx:matlabParser.Def_functionContext):
        pass

    # Exit a parse tree produced by matlabParser#def_function.
    def exitDef_function(self, ctx:matlabParser.Def_functionContext):
        pass


    # Enter a parse tree produced by matlabParser#def_class.
    def enterDef_class(self, ctx:matlabParser.Def_classContext):
        pass

    # Exit a parse tree produced by matlabParser#def_class.
    def exitDef_class(self, ctx:matlabParser.Def_classContext):
        pass


    # Enter a parse tree produced by matlabParser#attrib_class_boolean.
    def enterAttrib_class_boolean(self, ctx:matlabParser.Attrib_class_booleanContext):
        pass

    # Exit a parse tree produced by matlabParser#attrib_class_boolean.
    def exitAttrib_class_boolean(self, ctx:matlabParser.Attrib_class_booleanContext):
        pass


    # Enter a parse tree produced by matlabParser#attrib_class_meta.
    def enterAttrib_class_meta(self, ctx:matlabParser.Attrib_class_metaContext):
        pass

    # Exit a parse tree produced by matlabParser#attrib_class_meta.
    def exitAttrib_class_meta(self, ctx:matlabParser.Attrib_class_metaContext):
        pass


    # Enter a parse tree produced by matlabParser#attrib_property_boolean.
    def enterAttrib_property_boolean(self, ctx:matlabParser.Attrib_property_booleanContext):
        pass

    # Exit a parse tree produced by matlabParser#attrib_property_boolean.
    def exitAttrib_property_boolean(self, ctx:matlabParser.Attrib_property_booleanContext):
        pass


    # Enter a parse tree produced by matlabParser#attrib_property_access.
    def enterAttrib_property_access(self, ctx:matlabParser.Attrib_property_accessContext):
        pass

    # Exit a parse tree produced by matlabParser#attrib_property_access.
    def exitAttrib_property_access(self, ctx:matlabParser.Attrib_property_accessContext):
        pass


    # Enter a parse tree produced by matlabParser#attrib_method_boolean.
    def enterAttrib_method_boolean(self, ctx:matlabParser.Attrib_method_booleanContext):
        pass

    # Exit a parse tree produced by matlabParser#attrib_method_boolean.
    def exitAttrib_method_boolean(self, ctx:matlabParser.Attrib_method_booleanContext):
        pass


    # Enter a parse tree produced by matlabParser#attrib_method_access.
    def enterAttrib_method_access(self, ctx:matlabParser.Attrib_method_accessContext):
        pass

    # Exit a parse tree produced by matlabParser#attrib_method_access.
    def exitAttrib_method_access(self, ctx:matlabParser.Attrib_method_accessContext):
        pass


    # Enter a parse tree produced by matlabParser#atom_access.
    def enterAtom_access(self, ctx:matlabParser.Atom_accessContext):
        pass

    # Exit a parse tree produced by matlabParser#atom_access.
    def exitAtom_access(self, ctx:matlabParser.Atom_accessContext):
        pass


    # Enter a parse tree produced by matlabParser#st_assign.
    def enterSt_assign(self, ctx:matlabParser.St_assignContext):
        pass

    # Exit a parse tree produced by matlabParser#st_assign.
    def exitSt_assign(self, ctx:matlabParser.St_assignContext):
        pass


    # Enter a parse tree produced by matlabParser#st_command.
    def enterSt_command(self, ctx:matlabParser.St_commandContext):
        pass

    # Exit a parse tree produced by matlabParser#st_command.
    def exitSt_command(self, ctx:matlabParser.St_commandContext):
        pass


    # Enter a parse tree produced by matlabParser#st_if.
    def enterSt_if(self, ctx:matlabParser.St_ifContext):
        pass

    # Exit a parse tree produced by matlabParser#st_if.
    def exitSt_if(self, ctx:matlabParser.St_ifContext):
        pass


    # Enter a parse tree produced by matlabParser#st_for.
    def enterSt_for(self, ctx:matlabParser.St_forContext):
        pass

    # Exit a parse tree produced by matlabParser#st_for.
    def exitSt_for(self, ctx:matlabParser.St_forContext):
        pass


    # Enter a parse tree produced by matlabParser#st_switch.
    def enterSt_switch(self, ctx:matlabParser.St_switchContext):
        pass

    # Exit a parse tree produced by matlabParser#st_switch.
    def exitSt_switch(self, ctx:matlabParser.St_switchContext):
        pass


    # Enter a parse tree produced by matlabParser#st_try.
    def enterSt_try(self, ctx:matlabParser.St_tryContext):
        pass

    # Exit a parse tree produced by matlabParser#st_try.
    def exitSt_try(self, ctx:matlabParser.St_tryContext):
        pass


    # Enter a parse tree produced by matlabParser#st_while.
    def enterSt_while(self, ctx:matlabParser.St_whileContext):
        pass

    # Exit a parse tree produced by matlabParser#st_while.
    def exitSt_while(self, ctx:matlabParser.St_whileContext):
        pass


    # Enter a parse tree produced by matlabParser#function_params.
    def enterFunction_params(self, ctx:matlabParser.Function_paramsContext):
        pass

    # Exit a parse tree produced by matlabParser#function_params.
    def exitFunction_params(self, ctx:matlabParser.Function_paramsContext):
        pass


    # Enter a parse tree produced by matlabParser#function_returns.
    def enterFunction_returns(self, ctx:matlabParser.Function_returnsContext):
        pass

    # Exit a parse tree produced by matlabParser#function_returns.
    def exitFunction_returns(self, ctx:matlabParser.Function_returnsContext):
        pass


    # Enter a parse tree produced by matlabParser#statement.
    def enterStatement(self, ctx:matlabParser.StatementContext):
        pass

    # Exit a parse tree produced by matlabParser#statement.
    def exitStatement(self, ctx:matlabParser.StatementContext):
        pass


    # Enter a parse tree produced by matlabParser#xpr_tree.
    def enterXpr_tree(self, ctx:matlabParser.Xpr_treeContext):
        pass

    # Exit a parse tree produced by matlabParser#xpr_tree.
    def exitXpr_tree(self, ctx:matlabParser.Xpr_treeContext):
        pass


    # Enter a parse tree produced by matlabParser#xpr_tree_.
    def enterXpr_tree_(self, ctx:matlabParser.Xpr_tree_Context):
        pass

    # Exit a parse tree produced by matlabParser#xpr_tree_.
    def exitXpr_tree_(self, ctx:matlabParser.Xpr_tree_Context):
        pass


    # Enter a parse tree produced by matlabParser#xpr_array.
    def enterXpr_array(self, ctx:matlabParser.Xpr_arrayContext):
        pass

    # Exit a parse tree produced by matlabParser#xpr_array.
    def exitXpr_array(self, ctx:matlabParser.Xpr_arrayContext):
        pass


    # Enter a parse tree produced by matlabParser#xpr_array_.
    def enterXpr_array_(self, ctx:matlabParser.Xpr_array_Context):
        pass

    # Exit a parse tree produced by matlabParser#xpr_array_.
    def exitXpr_array_(self, ctx:matlabParser.Xpr_array_Context):
        pass


    # Enter a parse tree produced by matlabParser#xpr_cell.
    def enterXpr_cell(self, ctx:matlabParser.Xpr_cellContext):
        pass

    # Exit a parse tree produced by matlabParser#xpr_cell.
    def exitXpr_cell(self, ctx:matlabParser.Xpr_cellContext):
        pass


    # Enter a parse tree produced by matlabParser#xpr_cell_.
    def enterXpr_cell_(self, ctx:matlabParser.Xpr_cell_Context):
        pass

    # Exit a parse tree produced by matlabParser#xpr_cell_.
    def exitXpr_cell_(self, ctx:matlabParser.Xpr_cell_Context):
        pass


    # Enter a parse tree produced by matlabParser#xpr_array_index.
    def enterXpr_array_index(self, ctx:matlabParser.Xpr_array_indexContext):
        pass

    # Exit a parse tree produced by matlabParser#xpr_array_index.
    def exitXpr_array_index(self, ctx:matlabParser.Xpr_array_indexContext):
        pass


    # Enter a parse tree produced by matlabParser#xpr_cell_index.
    def enterXpr_cell_index(self, ctx:matlabParser.Xpr_cell_indexContext):
        pass

    # Exit a parse tree produced by matlabParser#xpr_cell_index.
    def exitXpr_cell_index(self, ctx:matlabParser.Xpr_cell_indexContext):
        pass


    # Enter a parse tree produced by matlabParser#xpr_field.
    def enterXpr_field(self, ctx:matlabParser.Xpr_fieldContext):
        pass

    # Exit a parse tree produced by matlabParser#xpr_field.
    def exitXpr_field(self, ctx:matlabParser.Xpr_fieldContext):
        pass


    # Enter a parse tree produced by matlabParser#xpr_function.
    def enterXpr_function(self, ctx:matlabParser.Xpr_functionContext):
        pass

    # Exit a parse tree produced by matlabParser#xpr_function.
    def exitXpr_function(self, ctx:matlabParser.Xpr_functionContext):
        pass


    # Enter a parse tree produced by matlabParser#xpr_handle.
    def enterXpr_handle(self, ctx:matlabParser.Xpr_handleContext):
        pass

    # Exit a parse tree produced by matlabParser#xpr_handle.
    def exitXpr_handle(self, ctx:matlabParser.Xpr_handleContext):
        pass


    # Enter a parse tree produced by matlabParser#command_argument.
    def enterCommand_argument(self, ctx:matlabParser.Command_argumentContext):
        pass

    # Exit a parse tree produced by matlabParser#command_argument.
    def exitCommand_argument(self, ctx:matlabParser.Command_argumentContext):
        pass



del matlabParser